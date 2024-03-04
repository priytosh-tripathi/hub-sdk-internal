import json
import logging
import sys
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Union


from hub_sdk.base.auth import Auth
from hub_sdk.modules.models import ModelList, Models

def require_authentication(func) -> callable:
    """
    A decorator function to ensure that the wrapped method can only be executed if the client is authenticated.

    Args:
        func (callable): The method to be wrapped.

    Returns:
        (callable): The wrapped method.
    """

    def wrapper(self, *args, **kwargs):
        """Decorator to ensure a method is called only if the user is authenticated."""
        if not self.authenticated and not kwargs.get("public"):
            raise PermissionError("Access Denied: Authentication required.")
        return func(self, *args, **kwargs)

    return wrapper

@dataclass
class DateGroup:
    train: Optional[dict] = None
    trained: Optional[datetime] = field(default=None, repr=lambda x: x.isoformat() if x else None)
    checkpoint: Optional[datetime] = field(default=None, repr=lambda x: x.isoformat() if x else None)

@dataclass
class ModelInput:
    # Fields filled based on ModelInputSchema
    id: str
    epochs: Optional[int] = None  # Defined when NOT using Ultralytics Cloud Timed Training
    imageSize: int
    patience: int
    cache: str
    device: Union[str, int, list[int]]
    batchSize: int
    advanced: dict = field(default_factory=dict)  # Assuming AdvancedModelConfiguration is a dict
    time: Optional[int] = None  # Defined when using Ultralytics Cloud Timed Training
    datasetId: str
    datasetPath: str
    datasetName: str
    projectId: str
    projectName: str
    parentId: str
    parentName: str
    parentArchitecture: str
    parentUrl: str
    parentPath: Optional[str] = None  # Defined when the parent is a custom pre-trained model
    name: str
    defaultName: str
    privacy: dict = field(default_factory=dict)  # Assuming Privacy is a dict
    roboflowType: str
    roboflowVersion: str

@dataclass
class NewModel(ModelInput):
    dates: DateGroup = field(default_factory=DateGroup)
    meta: dict = field(default_factory=dict)
    trainingMeta: dict = field(default_factory=dict)
    exports: Optional[dict] = None
    owner: dict = field(default_factory=dict)
    team: Optional[dict] = None
    config: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)
    project: dict = field(default_factory=dict)
    order: Optional[int] = None

    # ! Only in local state
    id: Optional[str] = None
    status: Optional[str] = None
    totals: Optional[dict] = None

class JsonReader:
    """
    Class to read JSON data from a file and create a Model object.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def read_data(self):
        try:
            with open(self.file_path, "r") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found: {self.file_path}")
            sys.exit(1)

    def create_model(self) -> NewModel:
        if self.data is None:
            return NewModel()

        model = NewModel(**self.data)

        # Process "dates" field
        dates = model.get("dates", {})
        model.dates = model.dates or DateGroup()  # Ensure `dates` is always a DateGroup instance
        train_data = dates.get("train", {})
        model.dates.train = model.dates.train or {}  # Ensure `train` is always a dict
        model.dates.train.update({
            "start": self._parse_timestamp(train_data.get("start")),
            "resume": self._parse_timestamp(train_data.get("resume")),
            "end": self._parse_timestamp(train_data.get("end")),
            "heartbeat": self._parse_timestamp(train_data.get("heartbeat")),
            "initialized": self._parse_timestamp(train_data.get("initialized")),
        })
        model.dates.trained = self._parse_timestamp(dates.get("trained"))
        model.dates.checkpoint = self._parse_timestamp(dates.get("checkpoint"))

        # Process deprecated and optional fields
        model.meta = model.meta or {}  # Ensure `meta` is always a dict
        model.meta["name"] = model.pop("name", None)
        model.project = model.project or {}  # Ensure `project` is always a dict
        model.project["id"] = model.pop("projectId", None)
        model.trainingMeta = model.trainingMeta or {}  # Ensure `trainingMeta` is always a dict
        model.trainingMeta["last_epoch"] = model.pop("last_epoch", None)
        model.trainingMeta.update(model.pop("trainingMeta", {}))

        # Handle missing optional fields
        model.trainingMeta.setdefault("lastMetricEpoch", None)
        model.trainingMeta.setdefault("lastCheckpointEpoch", None)
        model.trainingMeta.setdefault("agent", None)
        model.lineage = model.lineage or {}  # Ensure `lineage` is always a dict
        model.lineage["parent"] = model.lineage.get("parent", {})  # Ensure `parent` is always a dict
        model.lineage["parent"].setdefault("path", None)

        # Remove unused fields
        if "metricsId" in model.trainingMeta:
            del model.trainingMeta["metricsId"]

        # Ensure other fields are present (initially empty for optional fields)
        model.exports = model.exports or None
        model.owner = model.owner or {}  # Ensure `owner` is always a dict
        model.team = model.team or None
        model.config = model.config or {}  # Ensure `config` is always a dict
        model.dataset = model.dataset or {"id": "", "name": "", "filepath": ""}  # Ensure dataset has all required fields
        model.lineage = model.lineage or {"architecture": {}, "parent": {}}  # Ensure lineage has required sub-objects
        model.lineage["architecture"] = model.lineage.get("architecture", {"id": "", "name": ""})  # Ensure architecture has required fields
        model.lineage["parent"] = model.lineage.get("parent", {"id": "", "name": "", "url": ""})  # Ensure parent has required fields
        model.project = model.project or {"id": "", "name": ""}  # Ensure project has required fields
        model.order = model.order or None
        model.id = model.id or None
        model.status = model.status or None
        model.totals = model.totals or None

        return model
    

    def _parse_timestamp(self, timestamp_str):
        if timestamp_str:
            return datetime.fromisoformat(timestamp_str).astimezone(timezone.utc)
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python model_creator.py <json_file_path>")
        sys.exit(1)

    # Create reader objects
    reader = JsonReader(sys.argv[1])

    # Read JSON data and create Model object
    reader.read_data()
    new_model = reader.create_model()

    new_model_id = Models.create_model(new_model)

    logging.log("New Model created with id ", new_model_id)
    print("New model created with model ID: ", new_model_id)

    add_weights_path = input("Enter path for Weights file: ")

    print("Adding Weights to the Model ",new_model_id)
    weights_addition_response = Models.upload_model(ModelInput.epochs,add_weights_path)

    print("Weights Added to the Model")
    logging.log("Weights added to the Model ID",new_model_id)

    add_metrics_path = input("Enter Metrics Json file Path")
    
    

if __name__ == "__main__":
    main()
