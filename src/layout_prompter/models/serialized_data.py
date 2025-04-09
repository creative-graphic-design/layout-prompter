from typing import List

from pydantic import BaseModel


class Coordinates(BaseModel):
    left: int
    top: int
    width: int
    height: int

    def to_tuple(self):
        return (self.left, self.top, self.width, self.height)


class SerializedData(BaseModel):
    class_name: str
    coord: Coordinates


class SerializedOutputData(BaseModel):
    layouts: List[SerializedData]


class MultipleSerializedOutputData(BaseModel):
    contents: List[SerializedOutputData]
