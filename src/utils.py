from typing import List
from supervisely import Annotation
from supervisely.api.annotation_api import AnnotationInfo
from supervisely.imaging.color import rgb2hex
from supervisely.decorators.profile import timeit

import src.globals as g


ds_annotators = {}
ds_img_infos = {}
ds_ann_infos = {}


def wrap_in_tag(text, color):
    return f'<i class="zmdi zmdi-circle" style="margin-right: 3px; color: {rgb2hex(color)}"></i><span style="margin-right: 6px;">{text}</span>'


@timeit
def get_annotators(datasets_ids: List[int]):
    annotators = set()
    for dataset_id in datasets_ids:
        if dataset_id not in ds_annotators:
            ds_annotators[dataset_id] = set()
            dataset_info = g.all_datasets[dataset_id]
            ann_infos = g.api.annotation.get_list(dataset_id)
            for ann_info in ann_infos:
                ann = Annotation.from_json(
                    ann_info.annotation, g.project_metas[dataset_info.project_id]
                )
                for label in ann.labels:
                    ds_annotators[dataset_id].add(label.geometry.labeler_login)
        annotators = annotators.union(ds_annotators[dataset_id])
    return annotators


@timeit
def get_score(report: List[dict]):
    if "error" in report:
        return "Error"
    for metric in report:
        if metric["metric_name"] == "overall-score":
            if metric["class_gt"] == "" and metric["tag_name"] == "" and metric["image_gt_id"] == 0:
                return metric["value"]
    return 0


@timeit
def get_img_infos(project_id, dataset_id):
    global ds_img_infos
    if dataset_id == "-":
        datasets_ids = [ds.id for ds in g.all_datasets.values() if ds.project_id == project_id]
    else:
        datasets_ids = [dataset_id]
    for ds_id in datasets_ids:
        if ds_id not in ds_img_infos:
            ds_img_infos[ds_id] = g.api.image.get_list(ds_id)
    return [img for ds_id in datasets_ids for img in ds_img_infos[ds_id]]


@timeit
def get_ann_infos(project_id, dataset_id):
    global ds_ann_infos
    if dataset_id == "-":
        datasets_ids = [ds.id for ds in g.all_datasets.values() if ds.project_id == project_id]
    else:
        datasets_ids = [dataset_id]
    for ds_id in datasets_ids:
        if ds_id not in ds_ann_infos:
            ds_ann_infos[ds_id] = g.api.annotation.get_list(ds_id)
    return [ai for ds_id in datasets_ids for ai in ds_ann_infos[ds_id]]


@timeit
def get_common_images(first_img_infos, second_img_infos):
    second_img_name_to_idx = {img.name: i for i, img in enumerate(second_img_infos)}
    paired_infos = []
    for first_img in first_img_infos:
        if first_img.name in second_img_name_to_idx:
            paired_infos.append(
                (first_img, second_img_infos[second_img_name_to_idx[first_img.name]])
            )
    return [paired_info[0] for paired_info in paired_infos], [
        paired_info[1] for paired_info in paired_infos
    ]


@timeit
def get_images_ann_infos(img_infos, ann_infos):
    img_id_to_ann_info = {ann_info.image_id: ann_info for ann_info in ann_infos}
    return [img_id_to_ann_info[img_info.id] for img_info in img_infos]


@timeit
def get_common_ann_infos(
    first_img_infos, second_img_infos, first_all_ann_infos, second_all_ann_infos
):
    first = get_images_ann_infos(first_img_infos, first_all_ann_infos)
    second = get_images_ann_infos(second_img_infos, second_all_ann_infos)
    return first, second


@timeit
def filter_labels_by_user(
    first_ann_infos,
    second_ann_infos,
    first_meta,
    second_meta,
    first_login,
    second_login,
):
    for i, first_ann_info in enumerate(first_ann_infos):
        ann = Annotation.from_json(first_ann_info.annotation, first_meta)
        filtered_labels = [
            label for label in ann.labels if label.geometry.labeler_login == first_login
        ]
        filtered_tags = [tag for tag in ann.img_tags if tag.labeler_login == first_login]
        ann = ann.clone(labels=filtered_labels, img_tags=filtered_tags)
        first_ann_infos[i] = AnnotationInfo(
            image_id=first_ann_info.image_id,
            image_name=first_ann_info.image_name,
            annotation=ann.to_json(),
            created_at=first_ann_info.created_at,
            updated_at=first_ann_info.updated_at,
        )
    for i, second_ann_info in enumerate(second_ann_infos):
        ann = Annotation.from_json(second_ann_info.annotation, second_meta)
        filtered_labels = [
            label for label in ann.labels if label.geometry.labeler_login == second_login
        ]
        filtered_tags = [tag for tag in ann.img_tags if tag.labeler_login == second_login]
        ann = ann.clone(labels=filtered_labels, img_tags=filtered_tags)
        second_ann_infos[i] = AnnotationInfo(
            image_id=second_ann_info.image_id,
            image_name=second_ann_info.image_name,
            annotation=ann.to_json(),
            created_at=second_ann_info.created_at,
            updated_at=second_ann_info.updated_at,
        )


@timeit
def get_classes(ann_infos, meta):
    class_counts = {}
    for ann_info in ann_infos:
        ann = Annotation.from_json(ann_info.annotation, meta)
        ann_class_counts = ann.stat_class_count([c.name for c in meta.obj_classes])
        ann_class_counts.pop("total")
        for class_name, count in ann_class_counts.items():
            class_counts[class_name] = class_counts.get(class_name, 0) + count
    return set(class_name for class_name, count in class_counts.items() if count > 0)


@timeit
def get_class_matches(first_classes, second_classes):
    return {class_name: class_name for class_name in first_classes.intersection(second_classes)}


@timeit
def get_tags_whitelists(first_ann_infos, second_ann_infos, first_meta, second_meta):
    first_tag_whitelist = set()
    first_obj_tags_whitelist = set()
    for ann_info in first_ann_infos:
        ann = Annotation.from_json(ann_info.annotation, first_meta)
        for tag in ann.img_tags:
            first_tag_whitelist.add(tag.name)
        for tag in [t for label in ann.labels for t in label.tags]:
            first_obj_tags_whitelist.add(tag.name)
    second_tag_whitelist = set()
    second_obj_tags_whitelist = set()
    for ann_info in second_ann_infos:
        ann = Annotation.from_json(ann_info.annotation, second_meta)
        for tag in ann.img_tags:
            second_tag_whitelist.add(tag.name)
        for tag in [t for label in ann.labels for t in label.tags]:
            second_obj_tags_whitelist.add(tag.name)

    tags_whitelist = list(first_tag_whitelist.intersection(second_tag_whitelist))
    obj_tags_whitelist = list(first_obj_tags_whitelist.intersection(second_obj_tags_whitelist))
    return tags_whitelist, obj_tags_whitelist


def get_project_by_id(project_id):
    if project_id not in g.data["projects"]:
        g.data["projects"][project_id] = g.api.project.get_info_by_id(project_id)
    return g.data["projects"][project_id]


def get_project_by_name(workspace_id, project_name):
    for project in g.data["projects"].values():
        if project.name == project_name and project.workspace_id == workspace_id:
            return project
    project = g.api.project.get_info_by_name(workspace_id, project_name)
    if project is not None:
        g.data["projects"][project.id] = project
    return project


def get_dataset_by_id(dataset_id):
    if dataset_id not in g.data["datasets"]:
        g.data["datasets"][dataset_id] = g.api.dataset.get_info_by_id(dataset_id)
    return g.data["datasets"][dataset_id]


def get_dataset_by_name(project_id, dataset_name):
    for dataset in g.data["datasets"]:
        if dataset.name == dataset_name and dataset.project_id == project_id:
            return dataset
    dataset = g.api.dataset.get_info_by_name(project_id, dataset_name)
    if dataset is not None:
        g.data["datasets"][dataset.id] = dataset
    return dataset
