from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32
from datetime import timedelta

iris = Entity(
    name="entity_id",
    join_keys=["entity_id"],
)

iris_source = FileSource(
    path="../data/features/iris_features.csv",
    timestamp_field="event_timestamp",
)

iris_features_view = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=1),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    online=True,
    source=iris_source,
)

