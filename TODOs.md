# Datasets
## DONE
- ~~Add class `VisionDataset`~~
- ~~Make the method `set_root_dir` more understandable~~
- ~~In `to_csv`, modify the item so not to include the `root_dir` part~~
- ~~Remove the field `MEDIA_ID` from annotations~~
- ~~Do not change the field MEDIA_ID for ID in metadata~~
- ~~Implement method `__setitem__` on `Datasets`~~
- ~~Change the partitions management~~
- ~~Create two generic methods to assign data to `annotations` and `metadata` through the right index and use it, for example, to assign dimensions, bboxes and partitions~~.
- ~~In `_fit_metadata`, try to avoid re-processing all fields~~
- ~~Improve logic of internal updates when filtering (`filter_by_column`, use field `ANNOTATIONS.ID`)~~
- ~~Check all the dataset constructors calls and change most of them to DS.from_dataframe()~~
- ~~Avoid data validation when creating a dataset `copy`. Also avoid initializations of anns and meta~~
- ~~Check the logic of `set_field_values`~~. **Accept dataframes as values**
- ~~In method `create_crops_dataset`: Assign fields: image_id and id~~
- ~~Change the default value of the parameter `inplace` to `False` (verify the calls of all methods): set_field_values, filter_by_column, map_categories, filter_by_categories, filter_by_partition, filter_by_label_counts, filter_by_score, filter, sample~~
- ~~Implement method `draw_bounding_boxes`~~.
- Fix method `create_object_level_dataset_using_detections`
    - ~~assert that all `common_fld_anns_dets` values are present in `detections` and `anns`~~
    - ~~check if there are fields to remove from `obj_level_ds` (e.g., `image_id`)~~
- ~~Error in `to_csv`~~.
- ~~Refactor `Megadetector` class~~

## TODO
- Overload `+` operator:
    `def __add__(self, other): return type(self).from_datasets(self, other)`
- **Check empty datasets cases**
    - Add verification of `len(ds) == 0` when filtering
- *Update method `_get_dataframe_from_json` to adapt to changes in `media_id` field*
- Allow to have metadata entries with no annotations
- *Reduce the number of kwargs*
- When modifying the field 'item', validate consistency of root_dir
- When passing a `root_dir` relative to the current dir, validate how to store the items in CSVs
- Unify all the concepts (e.g., ~~columns, fields~~; categories, classes; etc.)
- Control when the logging messages should be silent
- Check if it is possible to rename the field ~video_id~ to `parent_video_id` for image datasets (e.g., in method `create_object_level_dataset_using_detections`)
- Fix the call from method `create_crops_dataset_using_detections`
- Use `__all__` to define elements to import when `import *`
- Remove column 'partition' of a dataset after applying `filter_by_partition`
- Research [supervision](https://github.com/roboflow/supervision)

# Models
- Refactor the method `classify_images_dataset`

# Evaluation
- Create the interfaces for evaluators and metrics
