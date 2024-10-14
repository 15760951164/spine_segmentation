
import numpy as np


class Landmark(object):

    def __init__(self,
                 coords=None,
                 is_valid=True,
                 scale=1.0,
                 value=0,
                 label=-1):

        self.coords = coords
        self.is_valid = is_valid
        if self.is_valid is None:
            self.is_valid = self.coords is not None
        self.scale = scale
        self.value = value
        self.label = label

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other) -> bool:
        return self.label == other.label


def get_mean_coords(landmarks):
    """
    Returns mean coordinates of a landmark list.
    :param landmarks: Landmark list.
    :return: np.array of mean coordinates.
    """
    valid_coords = [landmark.coords for landmark in landmarks if landmark.is_valid]
    return np.nanmean(np.stack(valid_coords, axis=0), axis=0)

def get_mean_landmark(landmarks):
    """
    Returns a Landmark object, where the coordinates are the mean coordinates of the
    given landmark list. scale and value are ignored.
    :param landmarks: Landmark list.
    :return: Landmark object with the mean coordinates.
    """
    valid_coords = [landmark.coords for landmark in landmarks if landmark.is_valid]
    return Landmark(np.nanmean(np.stack(valid_coords, axis=0), axis=0))

def get_mean_landmark_list(*landmarks):
    """
    Returns a list of mean Landmarks for two or more given lists of landmarks. The given lists
    must have the same length. The mean of corresponding list entries is calculated with get_mean_landmark.
    :param landmarks: Two or more given lists of landmarks.
    :return: List of mean landmarks.
    """
    return [get_mean_landmark(l) for l in zip(*landmarks)]
