import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path

from typing import Optional, Literal
from typing_extensions import Annotated
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError, conlist, ConfigDict

CONFIDENCE = Annotated[float, Field(ge=0.0, le=1.0)]

PII = Literal['faces', 'license_plates']
GEOLOCATION = Literal['synthetic', 'rtk', 'gcp', 'waas', 'gps', 'none']
CALIBRATION = Literal['synthetic', 'pix4d', 'webodm', 'agisoft', 'pnp', 'matlab', 'none']
ENV_CONDITIONS = Literal['clear', 'sunny', 'partly_cloudy', 'overcast', 'raining', 'wet_ground', 'snowing', 'snow_on_ground', 'other',
                         "siteM01:lighting1", "siteM01:lighting2", "siteM01:lighting3", "siteM01:lighting4", "siteM01:lighting5", "siteM01:lighting6",
                         "siteM01:lighting7", "siteM01:lighting8", "siteM01:lighting9"]
CAMERA_TYPE = Literal['ground', 'security', 'airborne', 'satellite']
CAMERA_PROJECTION = Literal['perspective', 'fisheye_pix4d']
CAMERA_MODES = Literal['ptz', 'night_imaging', 'thermal']
TRANSIENT_OCCLUSIONS = Literal['large', 'medium', 'small', 'negligible', 'information_overlay']
ARTIFACTS = Literal['saturation', 'lens_reflection_or_flare', 'dirty_lens', 'blur_or_defocus', 'compression', 'noise_or_grain', 'pixel_corruption', 'other_artifact']

class RepairedMetadata(BaseModel):
    # matches for version 1.1.x (x is any number)
    version: str = Field(pattern=r"^1\.1\.\d+$")
    # filename of corresponding input image
    fname: str
    # hour of day (0 - 23)
    time_of_day: Annotated[int, Field(ge=0, le=23)]
    # predicted confidence in time of day
    time_of_day_conf: CONFIDENCE
    # month of the year (1-12)
    time_of_year: Annotated[int, Field(ge=1, le=12)]
    # predicted confidence in time of year
    time_of_year_conf: CONFIDENCE
    # local cartesian x/y/z coordinates of camera, meters
    x: float
    y: float
    z: float
    # latitude/longitude/altitude of camera, WGS84 (meters for altitude)
    lat: float
    lon: float
    alt: float
    # predicted standard deviation/confidence of relative 3D camera location, meters
    geolocation_conf: float
    # global artifact boolean prediction
    artifact_pred: bool
    # global artifact prediction confidence
    artifact_pred_conf: CONFIDENCE
    # PTZ capability prediction
    ptz_pred: bool
    # PTZ capability confidence
    ptz_pred_conf: CONFIDENCE
    # nighttime imaging capability prediction
    nighttime_pred: bool
    # nighttime imaging capability confidence
    nighttime_pred_conf: CONFIDENCE

class CameraExtrinsics(BaseModel):
    # Latitude, degrees
    lat: float
    # Longitude, degrees
    lon: float
    # Altitutude, meters
    alt: float
    # Omega orientation angle, degrees
    omega: float
    # Phi orientation angle, degrees
    phi: float
    # Kappa orientation angle, degrees
    kappa: float


class CameraIntrinsics(BaseModel):
    # Focal length, pixels
    fx: float
    # Focal length, pixels
    fy: float
    # cx/cy: Optical center, pixels (OpenCV image coordinate system, where the origin, (0, 0), is located at the top-left of the image)
    cx: float
    cy: float
    # k1/k2/k3: Radial distortion coefficient of lens
    k1: float
    k2: float
    k3: float
    # p1/p2/p3/p4: Tangential distortion coefficient of lens
    p1: float
    p2: float
    p3: Optional[float] = None
    p4: Optional[float] = None
    # Affine transformation width, pixels
    C: Optional[float] = None
    # D/E/F Affine transformation parameter, pixels
    D: Optional[float] = None
    E: Optional[float] = None
    F: Optional[float] = None
    # Number of rows in image, pixels
    rows: float
    # Number of columns in image, pixels
    columns: float


class InputMetadata(BaseModel):
    # matches for version 4.2.x (x is any number)
    version: str = Field(pattern=r"^4\.2\.\d+$")
    # Filename of corresponding asset
    fname: str
    # Site ID in which the asset was collected
    site: str
    # Camera ID with which asset was collected
    source: str
    # Collection name in which asset was collected
    collection: str
    # POSIX timestamp at time of asset collection
    timestamp: float
    # See CameraExtrinsics class
    extrinsics: CameraExtrinsics
    # See CameraIntrinsics class
    intrinsics: CameraIntrinsics
    # Camera projection/model
    projection: CAMERA_PROJECTION
    # Enumeration value representing camera calibration method
    calibration: CALIBRATION
    # Enumeration value representing geolocation calibration method
    geolocation: GEOLOCATION
    # Status of PII detection/removal
    pii_status: str
    # List of enumeration values representing local weather/other environmental conditions, if applicable
    env_conditions: conlist(ENV_CONDITIONS)
    # Camera category
    type: CAMERA_TYPE
    # List of camera capabilities that were active while collecting the asset
    modes: conlist(CAMERA_MODES)
    # Bool representing exterior status of asset collection (ie, was the asset collected outdoors)
    exterior: bool
    # Bool representing interior status of asset collection (i.e., was the asset collected indoors
    interior: bool
    # List of transient occlusions in the asset
    transient_occlusions: conlist(TRANSIENT_OCCLUSIONS)
    # List of imaging artifacts detected in asset
    artifacts: conlist(ARTIFACTS)
    # Masks of transient occlusions in RLE format
    masks: dict
    # List of PII detected in the asset
    pii_detected: conlist(PII)
    # List of PII removed/blurred in the asset
    pii_removed: conlist(PII)


logging.captureWarnings(True)
logger = logging.getLogger("wrivalib.metadata_validator")


def configure_logger() -> None:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def load_metadata_json(path: Path) -> OrderedDict:
    """
    Loads JSON at `path` into an ordered dictionary.

    :param path: location on disk where json metadata is stored
    :return: ordered dict where the key is the filename and the value is the metadata
    """

    json_paths = []
    metadata = OrderedDict()

    if path.is_dir():
        json_paths = sorted(path.glob("**/*.json"))
        assert len(json_paths) != 0, f"No JSON found at {path}"
        logger.info(f"Found {len(json_paths)} metadata files to validate...")
    elif path.is_file():
        assert path.suffix == ".json", f"Invalid file type: {path.suffix}"
        logger.info(f"Loaded metadata file to validate...")
        json_paths.append(path)

    for json_path in tqdm(json_paths):
        with open(json_path) as f:
            metadata[json_path.name] = json.load(f)

    return metadata

def validate_repaired_collection(repaired_metadata: OrderedDict, strict=False) -> bool:
    """
    Validates a given collection of repaired metadata

    :param repaired_metadata: Ordered dict of repaired metadata as loaded from load_metadata_json
    :param strict: Whether or not Pydantic will be "strict"; strict=True will cause errors for incorrect types and
    extraneous information.
    :return: True if all metadata in the collection was valid; False if any one metadata was invalid
    """
    is_good = True

    for fname, asset in repaired_metadata.items():
        try:
            RepairedMetadata.model_validate(asset, strict=strict)
        except ValidationError as e:
            logger.error(f"Failed to validate repaired asset {fname}; see pydantic exception")
            logger.exception(e)
            is_good = False

    return is_good


def validate_input_collection(input_metadata: OrderedDict, strict=False) -> bool:
    """
    Validates a given collection of input metadata

    :param input_metadata: Ordered dict of input metadata as loaded from load_metadata_json
    :param strict: Whether or not Pydantic will be "strict"; strict=True will cause errors for incorrect types and
    extraneous information.
    :return: True if all metadata in the collection was valid; False if any one metadata was invalid
    """
    is_good = True

    for fname, asset in input_metadata.items():
        try:
            InputMetadata.model_validate(asset, strict=strict)
        except ValidationError as e:
            logger.error(f"Failed to validate input asset {fname}; see pydantic exception")
            logger.exception(e)
            is_good = False

    return is_good


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repaired",
        type=str,
        default="",
        help="Directory of the repaired metadata. Mutually exclusive with '--input' flag",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Directory of the input metadata. Mutually exclusive with the '--repaired' flag",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Whether or not Pydantic will be strict in it's model validation"
    )
    cmd_args = parser.parse_args()

    configure_logger()

    # assert they're indeed XOR
    assert bool(cmd_args.input) ^ bool(
        cmd_args.repaired
    ), "'--repaired' and '--input' flags are mutually exclusive"
    # assign directory based on xor condition
    metadata_dir = (
        Path(cmd_args.repaired) if not cmd_args.input else Path(cmd_args.input)
    )

    collection = load_metadata_json(metadata_dir)

    valid = (
        validate_input_collection(collection, strict=cmd_args.strict)
        if cmd_args.input
        else validate_repaired_collection(collection, strict=cmd_args.strict)
    )

    if valid:
        logger.info("Metadata was determined to be valid!")
        exit(0)
    else:
        logger.error(
            "Metadata was determined to be invalid. See log for more information."
        )
        exit(1)
