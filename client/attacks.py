from client.client_utils import estimate_static_RLU, estimated_entropy_from_grad, estimate_static_RLU_with_posterior
from client.client_utils import estimate_static_LLG,estimate_static_ZLG
from client.client_utils import create_synthetic_basis_matrix, calculate_distribution_ratios, calculate_dynamic_boost
import numpy as np
import torch
import copy
import scipy



def VLI_attack( gradients_for_prediction, args): 

    negative_elements = gradients_for_prediction[gradients_for_prediction < 0]
    
    if len(negative_elements) > 0:
        sum_negative = torch.sum(negative_elements)
    else:
        sum_negative = torch.sum(gradients_for_prediction)

    base_diagonal_val = sum_negative / (args.batch_size* args.local_epochs) 
    
    
    Basis = create_synthetic_basis_matrix(args.n_classes, base_diagonal_val)
    probs = calculate_distribution_ratios(gradients_for_prediction, Basis)
    max_p = np.max(probs)
    length = np.linalg.norm(gradients_for_prediction)


    
    
    
    boost_factor = calculate_dynamic_boost(probs, length, -base_diagonal_val)
    
    final_diagonal_val = base_diagonal_val * boost_factor
    Basis = create_synthetic_basis_matrix(args.n_classes, final_diagonal_val)

    residual = gradients_for_prediction.clone()
    counts = np.zeros(args.n_classes, dtype=int)
    
    for step in range(args.batch_size * args.local_epochs):
        scores = np.dot(residual, Basis)
        best_idx = np.argmax(scores)
        counts[best_idx] += 1
        
        projection_val = 1.0
        component_to_remove =   Basis[:, best_idx]
        residual = residual - component_to_remove



    return counts


def _project_to_simplex(vector, total):
    """Euclidean projection onto {x >= 0, sum(x) = total}."""
    vector = np.asarray(vector, dtype=np.float64)
    if total <= 0:
        raise ValueError("total must be positive")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - total
    candidates = ordered - cumulative / np.arange(1, len(vector) + 1) > 0
    rho = np.flatnonzero(candidates)[-1]
    theta = cumulative[rho] / (rho + 1)
    return np.maximum(vector - theta, 0.0)


def _integerize_counts(counts, total):
    """Round nonnegative real counts while preserving their integer total."""
    counts = _project_to_simplex(counts, total)
    result = np.floor(counts).astype(int)
    remainder = int(total - result.sum())
    if remainder:
        order = np.argsort(-(counts - result), kind="stable")
        result[order[:remainder]] += 1
    return result


def _as_numpy_vector(vector):
    if torch.is_tensor(vector):
        vector = vector.detach().cpu().numpy()
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("gradient vector must contain only finite values")
    return vector


def VLI_star_closed_form(gradients_for_prediction, args=None, n_samples=None,
                         return_diagnostics=False):
    """Closed-form VLI* under symmetric class difficulty and uniform confusion.

    The input must be the C-dimensional *bias-gradient* signal G, up to any
    positive common scale. The model assumptions imply at least one absent
    class and G_max > 0. The returned integer counts sum to n_samples.
    """
    signal = _as_numpy_vector(gradients_for_prediction)
    if n_samples is None:
        if args is None:
            raise ValueError("pass args or n_samples")
        n_samples = int(args.batch_size) * int(args.local_epochs)
    n_samples = int(n_samples)
    if n_samples <= 0 or signal.size < 2:
        raise ValueError("VLI* requires n_samples > 0 and at least two classes")

    g_max = float(np.max(signal))
    scale = max(float(np.max(np.abs(signal))), np.finfo(np.float64).tiny)
    if g_max <= np.finfo(np.float64).eps * scale * 16:
        raise ValueError(
            "closed-form VLI* needs a positive gradient component; this usually "
            "means no identifiable absent class or a near-zero signal"
        )

    raw = (n_samples / signal.size) * (1.0 - signal / g_max)
    counts = _integerize_counts(raw, n_samples)
    if return_diagnostics:
        return counts, {
            "raw_counts": raw,
            "g_max": g_max,
            "sum_gradient": float(signal.sum()),
            "projected": bool(np.any(raw < 0)),
        }
    return counts


def VLI_star_generalized(gradients_for_prediction, r, Q, n_samples,
                         return_diagnostics=False, tolerance=1e-8):
    """VLI* for known class difficulty r and confusion transition matrix Q.

    Convention: Q[j, k] is the fraction of class-j error mass sent to class k;
    Q is row-stochastic with zero diagonal. For irreducible Q, solve
    G = -(I - Q.T) a. Since this system has the Perron vector as its null
    direction, nonnegativity plus at least one absent class selects the unique
    boundary solution. Recover counts proportional to a / r.

    r and Q are oracle inputs in current experiments; this routine does not
    estimate them from the shared update.
    """
    signal = _as_numpy_vector(gradients_for_prediction)
    r = _as_numpy_vector(r)
    Q = np.asarray(Q, dtype=np.float64)
    n_samples = int(n_samples)
    classes = signal.size
    if classes < 2 or r.size != classes or Q.shape != (classes, classes):
        raise ValueError("G, r, and Q dimensions must agree and include >=2 classes")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not np.all(np.isfinite(Q)) or np.any(Q < -tolerance):
        raise ValueError("Q must be finite and nonnegative")
    if np.any(r <= 0) or not np.all(np.isfinite(r)):
        raise ValueError("all entries of r must be finite and positive")
    Q = np.maximum(Q, 0.0)
    if np.any(np.abs(np.diag(Q)) > tolerance):
        raise ValueError("Q must have a zero diagonal")
    row_sums = Q.sum(axis=1)
    if np.any(row_sums <= tolerance):
        raise ValueError("each Q row must have positive error mass")
    Q = Q / row_sums[:, None]

    # A unique positive Perron vector requires irreducibility. Without it,
    # the nullspace can have dimension > 1 and the update does not identify a.
    adjacency = Q > 0.0
    reach = adjacency.copy()
    np.fill_diagonal(reach, True)
    for k in range(classes):
        reach |= reach[:, k:k + 1] & reach[k:k + 1, :]
    if not np.all(reach):
        raise ValueError("Q must be irreducible for this generalized estimator")

    matrix = np.eye(classes) - Q.T
    a0, _, _, _ = np.linalg.lstsq(matrix, -signal, rcond=None)
    eigenvalues, eigenvectors = np.linalg.eig(Q.T)
    perron_idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    perron = np.real(eigenvectors[:, perron_idx])
    if perron.sum() < 0:
        perron = -perron
    if np.any(perron <= 0.0):
        raise ValueError("Q did not yield a strictly positive Perron vector")
    perron = perron / perron.sum()

    # Every solution is a0 + t*v. Select the smallest feasible t; by the
    # assumed absent class, this is the nonnegative solution on the boundary.
    t = float(np.max(-a0 / perron))
    a = a0 + t * perron
    a_scale = max(float(np.max(np.abs(a))), np.finfo(np.float64).tiny)
    a[np.abs(a) < tolerance * a_scale] = 0.0
    if np.min(a) < -tolerance * a_scale:
        raise ValueError("no nonnegative solution; supplied r/Q may not match G")
    a = np.maximum(a, 0.0)
    mass = a / r
    if mass.sum() <= np.finfo(np.float64).tiny:
        raise ValueError("the inferred class error masses are numerically zero")
    raw_counts = n_samples * mass / mass.sum()
    counts = _integerize_counts(raw_counts, n_samples)
    if return_diagnostics:
        residual = matrix @ a + signal
        return counts, {
            "a": a,
            "perron_vector": perron,
            "boundary_shift": t,
            "raw_counts": raw_counts,
            "relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(signal), np.finfo(np.float64).tiny)),
            "assumed_absent_classes": np.flatnonzero(a <= tolerance).tolist(),
        }
    return counts

def ZLGp_attack(model, gradients_for_prediction, O_bar, pj, aux_dataset, args):
    new_O_bar, new_pj = estimate_static_ZLG(args, copy.deepcopy(model), aux_dataset)
    new_O_bar = (new_O_bar + O_bar) / 2
    new_pj = (new_pj + pj) / 2


    n = []
    for i in range(args.n_classes):
        nj = args.batch_size * args.local_epochs * (new_pj[i].detach().cpu() - gradients_for_prediction[i] / new_O_bar.detach().cpu())
        n.append(max(int(nj.item()), 0))
    prop = (args.local_epochs * args.batch_size) / sum(n)
    for i in range(args.n_classes):
        n[i] = round(n[i] * prop)
    return n


def LLG_attack(model, gradients_for_prediction, args):
    

    h1_extraction = []
    negative_gradient = 0.0
    for i_cg, class_gradient in enumerate(gradients_for_prediction):
        if class_gradient < 0:
            h1_extraction.append((i_cg, class_gradient))
            negative_gradient += class_gradient
    
    new_impact = (1+1/args.n_classes)* (negative_gradient/ (args.batch_size))
    prediction = []

    for (i_c, _) in h1_extraction:
        prediction.append(i_c)
        gradients_for_prediction[i_c] = gradients_for_prediction[i_c].add(-new_impact)

    for _ in range(args.batch_size - len(prediction)):
        # add minimal candidate, likely to be doubled, to prediction
        min_id = torch.argmin(gradients_for_prediction).item()
        prediction.append(min_id)

        # add the mean value of one occurrence to the candidate
        gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-new_impact)

    n = []
    for i in range(args.n_classes):
        n.append(args.local_epochs * prediction.count(i))
    return n


def LLGp_attack(model, gradients_for_prediction, impact, offset, aux_dataset, args):
    new_impact, new_offset = estimate_static_LLG(args, copy.deepcopy(model), aux_dataset)
    new_impact = (new_impact + impact) / 2
    new_offset = (new_offset + offset) / 2

    h1_extraction = []

    for i_cg, class_gradient in enumerate(gradients_for_prediction):
        if class_gradient < 0:
            h1_extraction.append((i_cg, class_gradient))

    gradients_for_prediction -= new_offset
    prediction = []

    for (i_c, _) in h1_extraction:
        prediction.append(i_c)
        gradients_for_prediction[i_c] = gradients_for_prediction[i_c].add(-impact)

    for _ in range(args.batch_size - len(prediction)):
        # add minimal candidate, likely to be doubled, to prediction
        min_id = torch.argmin(gradients_for_prediction).item()
        prediction.append(min_id)

        # add the mean value of one occurrence to the candidate
        gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-new_impact)

    n = []
    for i in range(args.n_classes):
        n.append(args.local_epochs * prediction.count(i))
    return n
 

def rlu_rho_mean(args):
    """Mean of RLU's per-step coefficient rho(t), exactly as in the official per-scheme methods:
    FedProx -> Client_prox.RLU, otherwise -> Client.RLU (FedAvg; official code uses args.gamma here)."""
    rho = np.zeros(args.local_epochs)
    for t in range(args.local_epochs):
        if args.fl_scheme == 'fedprox':
            rho[t] = (1 - args.mu * args.lr) ** (args.local_epochs - 1 - t)
        else:
            rho[t] = (1 - args.gamma ** (args.local_epochs - t)) / (1 - args.gamma)
    return np.sum(rho) / args.local_epochs


def RLU_attack(model,w_grad_epochs, b_grad_epochs , latent_dim, mu, aux_dataset, args):
    new_mu, new_shift = estimate_static_RLU(args, model, aux_dataset)
    new_shift = scipy.special.softmax(new_mu)
    O = torch.zeros(latent_dim)
    
    for d in range(latent_dim):
        O[d] = torch.mean(w_grad_epochs[:, d] / b_grad_epochs)
    
    rho_mean = rlu_rho_mean(args)
    n = estimated_entropy_from_grad(args, new_shift*rho_mean, b_grad_epochs.detach().cpu().tolist(),
                                    args.batch_size * args.local_epochs)
    new_shift_softmax = estimate_static_RLU_with_posterior(args, n, mu, new_mu, O)
    n = estimated_entropy_from_grad(args, new_shift_softmax*rho_mean,b_grad_epochs.detach().cpu().tolist(), args.batch_size*args.local_epochs)
    return n
