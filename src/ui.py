import json
from typing import List
import supervisely as sly
from supervisely.app.widgets import Container, Text, Select, Button, Card, ConfusionMatrix
import src.globals as g
from src.metrics import calculate_exam_report


select_datasets_to_compare_items = [
    Select.Item(ds.id, f"{ds.name} (id: {ds.id}). Project: {g.all_projects[ds.project_id].name} (id: {g.all_projects[ds.project_id].id})") for ds in g.all_datasets.values()
]
select_datasets_to_compare = Select(items=select_datasets_to_compare_items, multiple=True)
compare_btn = Button("compare")
result_matrix = ConfusionMatrix()


def get_score(report: List[dict]):
    for metric in report:
        if metric["metric_name"] == "overall-score":
            if (
                metric["class_gt"] == ""
                and metric["tag_name"] == ""
                and metric["image_gt_id"] == 0
            ):
                return metric["value"]
    return 0


def get_matrix_data_json(first_ids, second_ids, pairs, pairs_scores):
    first_indexes = {id: idx for idx, id in enumerate(first_ids)}
    second_indexes = {id: idx for idx, id in enumerate(second_ids)}
    data = [[1 for _ in range(len(first_ids))] for _ in range(len(second_ids))]
    for pair, score in zip(pairs, pairs_scores):
        first_idx = first_indexes[pair[0]]
        second_idx = second_indexes[pair[1]]
        data[first_idx][second_idx] = round(score*100, 2)
        data[second_idx][first_idx] = round(score*100, 2)
    return {
        "columns": [f"{g.all_datasets[ds_id].name}. project: {g.all_projects[g.all_datasets[ds_id].project_id].name}" for ds_id in first_ids],
        "data": data,
    }


@compare_btn.click
def compare_btn_clicked():
    values = select_datasets_to_compare.get_value()
    pairs = [(values[i], values[j]) for i in range(len(values)) for j in range(i, len(values))]
    pairs_class_matches = []
    for pair in pairs:
        class_matches = []
        first = g.all_datasets[pair[0]]
        second = g.all_datasets[pair[1]]
        first_meta = g.project_metas[first.project_id]
        second_meta = g.project_metas[second.project_id]
        first_ann_infos = g.get_ann_infos(first.id)
        second_ann_infos = g.get_ann_infos(second.id)
        first_class_counts = {}
        for ann_info in first_ann_infos:
            ann = sly.Annotation.from_json(ann_info.annotation, first_meta)
            class_counts = ann.stat_class_count([c.name for c in first_meta.obj_classes])
            class_counts.pop("total")
            for class_name, count in class_counts.items():
                first_class_counts[class_name] = first_class_counts.get(class_name, 0) + count
        second_class_counts = {}
        for ann_info in second_ann_infos:
            ann = sly.Annotation.from_json(ann_info.annotation, second_meta)
            class_counts = ann.stat_class_count([c.name for c in second_meta.obj_classes])
            class_counts.pop("total")
            for class_name, count in class_counts.items():
                second_class_counts[class_name] = second_class_counts.get(class_name, 0) + count
        for class_name, count in first_class_counts.items():
            if class_name in second_class_counts:
                if count != 0 and second_class_counts[class_name] != 0:
                    class_matches.append({"class_gt": class_name, "class_pred": class_name})
        pairs_class_matches.append(class_matches)

    pairs_tags_whitelist = []
    pairs_obj_tags_whitelist = []
    for pair in pairs:
        first = g.all_datasets[pair[0]]
        second = g.all_datasets[pair[1]]
        first_meta = g.project_metas[first.project_id]
        second_meta = g.project_metas[second.project_id]
        first_ann_infos = g.get_ann_infos(first.id)
        second_ann_infos = g.get_ann_infos(second.id)
        first_tag_whitelist = set()
        first_obj_tags_whitelist = set()
        for ann_info in first_ann_infos:
            ann = sly.Annotation.from_json(ann_info.annotation, first_meta)
            for tag in ann.img_tags:
                first_tag_whitelist.add(tag.name)
            for tag in [t for label in ann.labels for t in label.tags]:
                first_obj_tags_whitelist.add(tag.name)
        second_tag_whitelist = set()
        second_obj_tags_whitelist = set()
        for ann_info in second_ann_infos:
            ann = sly.Annotation.from_json(ann_info.annotation, second_meta)
            for tag in ann.img_tags:
                second_tag_whitelist.add(tag.name)
            for tag in [t for label in ann.labels for t in label.tags]:
                second_obj_tags_whitelist.add(tag.name)

        pairs_tags_whitelist.append(list(first_tag_whitelist.intersection(second_tag_whitelist)))
        pairs_obj_tags_whitelist.append(list(first_obj_tags_whitelist.intersection(second_obj_tags_whitelist)))

    pair_reports = []
    for pair, class_matches, tags_whitelist, obj_tags_whitelist in zip(pairs, pairs_class_matches, pairs_tags_whitelist, pairs_obj_tags_whitelist):
        report = calculate_exam_report(
            server_address=g.server_address,
            api_token=g.api_token,
            project_gt_id=g.all_datasets[pair[0]].project_id,
            dataset_gt_id=pair[0],
            project_pred_id=g.all_datasets[pair[1]].project_id,
            dataset_pred_id=pair[1],
            class_matches=class_matches,
            tags_whitelist=tags_whitelist,
            obj_tags_whitelist=obj_tags_whitelist,
            iou_threshold=0.5,
        )
        pair_reports.append(report)

    pair_scores = [get_score(report) for report in pair_reports]

    result_matrix.read_json(get_matrix_data_json(values, values, pairs, pair_scores))


layout = Container(widgets=[
    Text("Consensus"),
    Card(title="Select Datasets to compare", content=select_datasets_to_compare),
    compare_btn,
    Card(title="Compare Results", content=result_matrix)
])
