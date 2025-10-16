import gpytorch
import pandas as pd
import torch
from typing import List, Union, Callable
from helpers.gp_classes import DataGPModel
from helpers.util_functions import prior_distribution, extract_model_parameters, reparameterize_model

# Registry for input patterns
INPUT_PATTERNS = {}

# Registry for label patterns
LABEL_PATTERNS = {}


def register_input_pattern(name: str):
    """
    Decorator to register input patterns.
    """
    def wrapper(func):
        INPUT_PATTERNS[name] = func
        return func
    return wrapper


def register_label_pattern(name: str):
    """
    Decorator to register label patterns.
    """
    def wrapper(func):
        LABEL_PATTERNS[name] = func
        return func
    return wrapper


def list_available_patterns():
    return {
        "input_patterns": list(INPUT_PATTERNS.keys()),
        "label_patterns": list(LABEL_PATTERNS.keys())
    }

# -------------------------------------------------------
# Registering input patterns
# -------------------------------------------------------


@register_input_pattern("linear shifted")
def linear_shifted(n_points: int, n_dim: int, **kwargs) -> torch.tensor:
    """
    Linear input pattern shifted by a constant value to the right.
    Will result in a linear pattern with a constant offset to the right by "SHIFT".

    Parameters
    ----------
    n_points : int
        Number of points to generate
    n_dim : int
        Number of dimensions
    kwargs : dict
        Additional arguments
            - START : float
                Base start value for the linear pattern, BEFORE shifting
            - END : float
                Base end value for the linear pattern, BEFORE shifting
            - SHIFT : float
                Shift value for the linear pattern
            - NOISE : float
                Noise value to add to the linear pattern    
            - dim_weights : list of floats
                Weights for each dimension (default: [1.0] * n_dim)

    Returns
    -------
    torch.tensor
        Generated input pattern

    """
    START = kwargs["START"] if "START" in kwargs else 0.0
    END = kwargs["END"] if "END" in kwargs else 1.0
    SHIFT = kwargs["SHIFT"] if "SHIFT" in kwargs else 0.0
    NOISE = kwargs["NOISE"] if "NOISE" in kwargs else 0.0
    # Purpose of dim weighting is to have dimensions grow at different rates
    dim_weights = kwargs["dim_weights"] if "dim_weights" in kwargs else [1.0] * n_dim
    # Example: [1.0, 2.0] means that the first dimension grows at 1x and the second at 2x
    base_data = torch.stack([torch.linspace(START, END, n_points) + SHIFT for _ in range(n_dim)], dim=-1)
    if n_dim > 1:
        # Apply dimension weights
        for i in range(n_dim):
            base_data[:, i] = base_data[:, i] * dim_weights[i]
    return base_data


@register_input_pattern("linear")
def linear(n_points: int, n_dim: int, **kwargs) -> torch.tensor:
    """
    Linear input pattern

    Parameters
    ----------
    n_points : int
        Number of points to generate
    n_dim : int
        Number of dimensions
    kwargs : dict
        Additional arguments
            - START : float
                Start value for the linear pattern
            - END : float
                End value for the linear pattern
            - NOISE : float
                Noise value for the linear pattern
            - dim_weights : list of floats
                Weights for each dimension (default: [1.0] * n_dim)

    Returns
    -------
    torch.tensor
        Generated input pattern

    """
    START = kwargs["START"] if "START" in kwargs else 0.0
    END = kwargs["END"] if "END" in kwargs else 1.0
    NOISE = kwargs["NOISE"] if "NOISE" in kwargs else 0.0
    # Purpose of dim weighting is to have dimensions grow at different rates
    dim_weights = kwargs["dim_weights"] if "dim_weights" in kwargs else [1.0] * n_dim
    # Example: [1.0, 2.0] means that the first dimension grows at 1x and the second at 2x
    base_data = torch.stack([torch.linspace(START, END, n_points) for _ in range(n_dim)], dim=-1)
    if n_dim > 1:
        # Apply dimension weights
        for i in range(n_dim):
            base_data[:, i] = base_data[:, i] * dim_weights[i]
    return base_data


# -------------------------------------------------------
# Registering label patterns
# -------------------------------------------------------

@register_label_pattern("periodic_1D")
def periodic_1D(X):
    """
    $\\sin(x_0)$
    """
    return torch.sin(X[:,0])

@register_label_pattern("periodic_2D")
def periodic_2D(X):
    """
    $\\sin(x_0) \\cdot \\sin(x_1)$
    """
    return torch.sin(X[:,0]) * torch.sin(X[:,1])

@register_label_pattern("parabola_1D")
def parabola_1D(X):
    """
    $x_0^2$
    """
    return X[:,0]**2

@register_label_pattern("parabola_2D")
def parabola_2D(X):
    """
    $x_0^2 \\cdot x_1^2$
    """
    return X[:,0]**2 + X[:,1]**2

@register_label_pattern("product")
def product(X):
    """
    $x_0 \\cdot x_1$
    """
    return X[:,0] * X[:,1]

@register_label_pattern("periodic_sum")
def periodic_sum(X):
    """
    $\\sin(x_0 + x_1)$
    """
    return torch.sin(X[:,0] + X[:,1])

@register_label_pattern("periodic_sincos")
def periodic_sincos(X):
    """
    $\\sin(x_0) \\cdot \\cos(x_1)$
    """
    return torch.sin(X[:,0]) * torch.cos(X[:,1])


@register_label_pattern("linear_1D")
def linear_1D(X):
    """
    $x_0$
    """
    return X[:,0]

@register_label_pattern("linear_2D")
def linear_2D(X):
    """
    $x_0 + x_1$
    """
    return X[:,0]+X[:,1]



class Transformations:
    """
    Transformations are functions that can be applied to the inputs or labels.
    """

    def __init__(self):
        pass

    @staticmethod
    def z_score(x: torch.tensor, **kwargs) -> torch.tensor:
        return_factors = kwargs["return_factors"] if "return_factors" in kwargs else False
        if return_factors:
            mean = x.mean()
            std = x.std()
            return (x - mean) / std, mean, std
        else:
            # Standardize the data
            return (x - x.mean()) / x.std()

    @staticmethod
    def inverse_z_score(x: torch.tensor, mean: torch.tensor, std: torch.tensor) -> torch.tensor:
        return x * std + mean


class LabelGenerator:
    """
    Gaussian Process patterns begin with "GP_".
    """

    def __init__(self, pattern : Union[str, Callable]):
        self.pattern = pattern
        pass


    def generate_labels(self, inputs : torch.tensor) -> torch.tensor:
        if callable(self.pattern):
            return self.pattern(inputs)
        else:
            if self.pattern.startswith("GP_"):
                # Generate a Gaussian Process pattern
                callable_pattern = self.generate_gp_callable_pattern(self.pattern[3:], inputs)
            else:
                # Generate a standard pattern
                callable_pattern = LABEL_PATTERNS[self.pattern]
            return callable_pattern(inputs)

    
    def generate_gp_callable_pattern(self, pattern : str, inputs : torch.tensor) -> Callable:
        n_dim = inputs.shape[1]
        alibi_x_points = torch.stack([torch.linspace(0, 1, 1) for _ in range(n_dim)], dim=-1)
        alibi_y_points = torch.linspace(0, 1, 1)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if n_dim == 1:
            gp = gp_classes.DataGPModel(alibi_x_points, alibi_y_points,likelihood, kernel_text=pattern)
        elif n_dim == 2:
            gp = gp_classes.DataMIGPModel(alibi_x_points, alibi_y_points,likelihood, kernel_text=pattern)

        def gp_callable(inputs):
            gp.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.prior_mode(True):
                gp_output = gp(inputs)
                labels = gp_output.mean
                return labels
        return gp_callable





def load_csv_tensor(filepath: str, target_dim=None, header=None, expected_ndim: int = 1) -> torch.Tensor:
    data = torch.tensor(pd.read_csv(filepath, header=header).values)
    
    if target_dim is not None:
        if isinstance(target_dim, list):
            data = data[:, target_dim]
        else:
            data = data[:, target_dim]
    elif data.ndim > expected_ndim:
        data = data[:, 0]
        print(f"Warning: data has more than {expected_ndim} dimensions. Using the first dimension.")

    return data


class DataGenerator:

    def __init__(self):
        pass
    
    def generate_inputs(self, pattern: Union[str, Callable], n_points: int = 0, n_dim: int = 1, **kwargs) -> torch.tensor:
        """
        Pattern might be read from a file or be a lambda expression
        """
        if isinstance(pattern, str):
            if pattern.endswith(".csv"):
                target_dim = kwargs["target_dim"] if "target_dim" in kwargs else None
                header = kwargs["header"] if "header" in kwargs else None
                input = load_csv_tensor(pattern, target_dim=target_dim, header=header, expected_ndim=n_dim)
            else:
                standard_pattern = INPUT_PATTERNS[pattern]
                input = standard_pattern(n_points, n_dim, **kwargs)
        if n_dim == 1:
            input = input.flatten()
        return input


    def generate_labels(self, inputs : torch.tensor=None,  pattern : Union[str, Callable]=None, **kwargs) -> torch.tensor:
        if isinstance(pattern, str):
            if pattern.endswith(".csv"):
                target_dim = kwargs["target_dim"] if "target_dim" in kwargs else None
                header = kwargs["header"] if "header" in kwargs else None
                n_dim = kwargs["n_dim"] if "n_dim" in kwargs else 1
                labels = load_csv_tensor(pattern, target_dim=target_dim, header=header, expected_ndim=n_dim)
            else:
                pattern = LabelGenerator(pattern)
                labels = pattern.generate_labels(inputs)
        else:
            # If pattern is a function, it should be callable
            labels = pattern(inputs)

        return labels


    def apply_transformations(self, inputs: torch.tensor, transformations: List[Callable] = None) -> torch.tensor:
        if not transformations:
            return inputs
        for transformation in transformations:
            inputs = transformation(inputs)
        return inputs


def nearest_idx_torch(a, b):
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    # pairwise distances for 1D values
    idx = (a[:, None] - b[None, :]).abs().argmin(dim=1)
    all_idx = torch.arange(b.shape[0], device=b.device)
    remaining_b = b[~torch.isin(all_idx, idx)]
    return idx, ~torch.isin(all_idx, idx), b[idx], remaining_b


def generate_appended_prediction_data(data_model, data_likelihood, train_obs, all_observations_y, eval_obs, test_dataset_count=10):
    total_obs = torch.cat([train_obs, eval_obs], dim=0)

    # Store if model is in training mode
    previous_state = data_model.training

    # Get into evaluation (predictive posterior) mode
    data_model.eval()
    data_likelihood.eval()

    all_test_observations_y = []
    for i in range(len(all_observations_y)):
        data_model.set_train_data(train_obs, all_observations_y[i], strict=False)
        with torch.no_grad():
            #f_preds = data_likelihood(data_model(eval_obs))
            f_preds = data_model(eval_obs)
        test_observations_y = f_preds.sample_n(test_dataset_count)
        all_test_observations_y.append(test_observations_y)

    # Reverse to previous state
    if previous_state:
        data_model.train()
        data_likelihood.train()

    return eval_obs, all_test_observations_y


def sample_data_from_gp(train_START, train_END, train_COUNT, data_kernel, eval_START=None, eval_END=None, eval_COUNT=100, train_dataset_count=5, test_data=True, test_dataset_count=10, interleaved_to_appended_ratio=0.0, use_interleaved_data_as_train_for_app_eval=False):
    if test_data: 
        # I want test data, but the interleaved ratio is smaller than 1.0 and there is no eval_START and no eval_END
        if interleaved_to_appended_ratio < 1.0 and (eval_START is None or eval_END is None):
            raise ValueError("If you want appended test data, you need to provide eval_START and eval_END.")
        

    # training data for model initialization (e.g. 1 point with x=0, y=0) ; this makes initializing the model easier
    prior_x = torch.linspace(0, 1, 1)
    prior_y = prior_x

    # initialize likelihood and model
    data_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    data_model = DataGPModel(prior_x, prior_y, data_likelihood, kernel_name=data_kernel)

    int_eval_COUNT = int(interleaved_to_appended_ratio * eval_COUNT)
    app_eval_COUNT = eval_COUNT - int_eval_COUNT

    # The interleaved data
    total_obs = torch.linspace(train_START, train_END, train_COUNT + int_eval_COUNT)
    train_obs = torch.linspace(train_START, train_END, train_COUNT)

    train_idx, int_eval_idx, train_obs, int_eval_obs = nearest_idx_torch(train_obs, total_obs)
    assert len(train_obs) == train_COUNT, "Something went wrong with the interleaved training data sampling."
    assert len(int_eval_obs) == int_eval_COUNT, "Something went wrong with the interleaved evaluation data sampling."
    assert len(int_eval_obs) + len(train_obs) == len(total_obs), "Something went wrong with the interleaved data sampling."

    # Get into evaluation (predictive posterior) mode
    data_model.eval()
    data_likelihood.eval()

    # Make predictions by feeding model
    with torch.no_grad(), gpytorch.settings.prior_mode(True):
        all_obs_preds = data_model(total_obs)

    all_observations_y = all_obs_preds.sample_n(max(train_dataset_count, test_dataset_count))
    all_int_eval_observations_y = all_observations_y[:, int_eval_idx]
    all_train_observations_y = all_observations_y[:, train_idx]

    if test_data is False:
        return (train_obs, all_train_observations_y), (None, None), (None, None)

    # The appended data
    if interleaved_to_appended_ratio == 1.0:
        return (train_obs, all_train_observations_y), (int_eval_obs, all_int_eval_observations_y), (None, None)
    eval_obs = torch.linspace(eval_START, eval_END, app_eval_COUNT)
  # Set the nosie to be almost zero to have the prediction samples continuously connected to the generated training data
    model_params = extract_model_parameters(data_model)
    model_params_bak = model_params.clone()
    model_params[0] = -10.0 # 
    reparameterize_model(data_model, model_params)

    # Use the interleaved data to further "constrain" the predictions for the appended test data
    if use_interleaved_data_as_train_for_app_eval:
        app_eval_obs, all_app_test_observations_y = generate_appended_prediction_data(data_model, data_likelihood, total_obs, all_observations_y=all_observations_y, eval_obs=eval_obs, test_dataset_count=test_dataset_count)
    # Or use only the training data
    else:
        app_eval_obs, all_app_test_observations_y = generate_appended_prediction_data(data_model, data_likelihood, train_obs, all_observations_y=all_train_observations_y, eval_obs=eval_obs, test_dataset_count=test_dataset_count)

    reparameterize_model(data_model, model_params_bak)

    if interleaved_to_appended_ratio == 0.0:
        return (train_obs, all_train_observations_y), (None, None), (app_eval_obs, all_app_test_observations_y)

    return (train_obs, all_train_observations_y), (int_eval_obs, all_int_eval_observations_y), (app_eval_obs, all_app_test_observations_y)