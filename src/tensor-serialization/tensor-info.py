import torch
import sys
import pickle


def get_tensor_info(t: torch.Tensor):
    """
    Computes and returns information about the size and data of a given tensor.

    This function calculates the size of the tensor's data in bytes, the serialized
    (pickled) representation of the tensor, and the raw data stored in the tensor.

    Parameters:
    t (torch.Tensor): The tensor to analyze.

    Returns:
    tuple:
        - int: Size of the tensor's data in bytes.
        - bytes: The pickled (serialized) tensor.
        - list: The raw data stored in the tensor.

    Raises:
    AssertionError: If there's a discrepancy in the calculated tensor size.
    """

    t_data_size = t.element_size() * t.nelement()

    # Validate the size using untyped storage
    assert t.untyped_storage().nbytes(
    ) == t_data_size, "There is a problem with the tensor type"

    # Total size including Python overhead
    total_size = sys.getsizeof(t) + t_data_size

    print(f'Tensor shape: {t.shape}')
    print(f'Tensor (data only) size: {t_data_size} bytes')
    print(f'Total tensor size (including Python overhead): {total_size} bytes')

    pickled_t = pickle.dumps(t)

    # Extract raw data from the tensor
    t_data = t.tolist()

    return t_data_size, pickled_t, t_data
