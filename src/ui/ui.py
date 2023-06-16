import json
from typing import Dict, List
import supervisely as sly
from supervisely.app.widgets import (
    Text,
    Container,
    Flexbox,
    Select,
    Button,
    Card,
    RadioTable,
    Field,
    Table,
    ProjectThumbnail,
    DatasetThumbnail,
    NotificationBox,
    Progress,
)
from supervisely.api.annotation_api import AnnotationInfo
from supervisely.imaging.color import rgb2hex

import src.globals as g
from src.ui.report import render_report, layout as report_layout
from src.metrics import calculate_exam_report


COMPARE_TABLE_COLUMNS = [
    "x",
    "Project",
    "Project id",
    "Dataset",
    "Dataset id",
    "Annotator",
]

ALL_DATASETS = "All datasets"
ALL_USERS = "All users"


class ComparisonResult:
    def __init__(
        self,
        pair,
        first_meta: sly.ProjectMeta,
        second_meta: sly.ProjectMeta,
        first_images,
        second_images,
        first_annotations_jsons,
        second_annotations_jsons,
        tags,
        classes,
        first_classes,
        second_classes,
        report,
        differences,
    ):
        self.pair = pair
        self.first_meta = first_meta
        self.second_meta = second_meta
        self.first_images = first_images
        self.second_images = second_images
        self._first_annotations_path = self.save_anns_jsons(
            ', '.join(str(p) for p in pair[0][2:]), first_annotations_jsons
        )
        self._second_annotations_path = self.save_anns_jsons(
            ', '.join(str(p) for p in pair[1][2:]), second_annotations_jsons
        )
        self.tags = tags
        self.classes = classes
        self.first_classes = first_classes
        self.second_classes = second_classes
        self._report_path = self.save_report(pair, report)
        self._differences_path = self.save_differences(pair, differences)
        try:
            self.error_message = report["error"]
        except (KeyError, TypeError):
            self.error_message = None

    def save_differences(self, pair, difference_geometries: List[sly.Bitmap]):
        first = ', '.join(str(p) for p in pair[0][2:])
        second = ', '.join(str(p) for p in pair[1][2:])
        path = f"/tmp/diffs_{first}{second}.json"
        with open(path, "w") as f:
            json.dump(
                [
                    None if geometry is None else geometry.to_json()
                    for geometry in difference_geometries
                ],
                f,
            )
        return path

    def save_anns_jsons(self, filename, anns: List[Dict]):
        path = f"/tmp/anns_{filename}.json"
        with open(path, "w") as f:
            json.dump(anns, f)
        return path

    def save_report(self, pair, report):
        first = ', '.join(str(p) for p in pair[0][2:])
        second = ', '.join(str(p) for p in pair[1][2:])
        path = f"/tmp/report_{first}{second}.json"
        with open(path, "w") as f:
            json.dump(report, f)
        return path

    def get_first_annotations(self):
        with open(self._first_annotations_path, "r") as f:
            return [
                sly.Annotation.from_json(ann_json, self.first_meta)
                for ann_json in json.load(f)
            ]

    def get_second_annotations(self):
        with open(self._second_annotations_path, "r") as f:
            return [
                sly.Annotation.from_json(ann_json, self.second_meta)
                for ann_json in json.load(f)
            ]

    def get_differences(self):
        with open(self._differences_path, "r") as f:
            return [
                None if bitmap_json is None else sly.Bitmap.from_json(bitmap_json)
                for bitmap_json in json.load(f)
            ]

    def get_report(self):
        with open(self._report_path, "r") as f:
            return json.load(f)


def wrap_in_tag(text, color):
    return f'<i class="zmdi zmdi-circle" style="margin-right: 3px; color: {rgb2hex(color)}"></i><span style="margin-right: 6px;">{text}</span>'


class SelectedUser:
    def __init__(self):
        self._project_name = None
        self._dataset_name = None
        self._user_login = None
        self._classes = None
        self._project_text = Text()
        self._dataset_text = Text()
        self._user_text = Text()
        self._classes_text = Text()
        self.layout = Flexbox(
            widgets=[
                Field(title="Project", content=self._project_text),
                Field(title="Dataset", content=self._dataset_text),
                Field(title="User", content=self._user_text),
                Field(title="Classes", content=self._classes_text),
            ],
            gap=30,
        )

    def update(self):
        self._project_text.set(self._project_name, status="text")
        self._dataset_text.set(self._dataset_name, status="text")
        self._user_text.set(self._user_login, status="text")
        self._classes_text.set(''.join([wrap_in_tag(class_name, color) for class_name, color in self._classes]), status="text")

    def set(self, project_name, dataset_name, user_login, classes):
        self._project_name = project_name
        self._dataset_name = dataset_name
        self._user_login = user_login
        self._classes = classes
        self.update()


workspace_thumbnail = Container(
    widgets=[
        Field(title="Team", description="", content=Text(g.team.name)),
        Field(
            title="Workspace",
            description="To change the workspace you need to run the application from desired workspace",
            content=Text(g.workspace.name),
        ),
    ]
)
select_project_to_compare_items = [
    Select.Item(project.id, project.name) for project in g.all_projects.values()
]
select_project_to_compare = Select(items=select_project_to_compare_items)
select_project_to_compare_field = Field(
    title="Project", content=select_project_to_compare
)
select_dataset_to_compare = Select(items=[Select.Item(None, "Combine all datasets")])
select_dataset_to_compare_field = Field(
    title="Dataset", content=select_dataset_to_compare
)
select_user_to_compare = Select(items=[Select.Item(None, "All users")])
add_to_compare_btn = Button("add", icon="zmdi zmdi-plus", button_size="small")
add_to_compare_btn.disable()
compare_table = RadioTable(columns=COMPARE_TABLE_COLUMNS, rows=[])
pop_row_btn = Button("remove", button_size="small")
compare_btn = Button("calculate consensus")
report_pairs_progress = Progress("Calculating consensus report...", show_percents=True, hide_on_finish=False)
report_pairs_progress.hide()
repot_preparation_progress = Progress("Preparing data for the report...", show_percents=True, hide_on_finish=False)
repot_preparation_progress.hide()
report_calculation_progress = Progress("Calculating consensus report for pair: ", show_percents=True, hide_on_finish=False)
report_calculation_progress.hide()
result_table = Table()
consensus_report_text = Text(f"<h1>Consensus report</h1>", status="text")
consensus_report_text.hide()
selected_pair_first = SelectedUser()
selected_pair_second = SelectedUser()
consensus_report_classes = Text()
consensus_report_details = Card(
    title="Details",
    description="Report is calculated for classes that are present in both sets of annotations",
    content=Container(
        widgets=[
            Flexbox(widgets=[Text("<h3>First:</h3>"), selected_pair_first.layout], gap=42),
            Flexbox(
                widgets=[Text("<h3>Second:</h3>"), selected_pair_second.layout], gap=20
            ),
            Flexbox(widgets=[Text("<b>Report Classes:</b>"), consensus_report_classes], gap=15),
        ]
    ),
)
consensus_report_details.hide()
consensus_report_notification = NotificationBox(
    title="Consensus report",
    description="Click on compare table cell to show a detailed consensus report for the pair",
    box_type="info",
)
consensus_report_notification.hide()
consensus_report_error_notification = NotificationBox(
    title="Error",
    description="Error occured while calculating consensus report",
    box_type="error",
)
consensus_report_error_notification.hide()
result_table.hide()
report_layout.hide()
pairs_comparisons_results = {}
name_to_row = {}


@sly.timeit
def get_annotators(datasets_ids: List[int]):
    annotators = set()
    for dataset_id in datasets_ids:
        dataset_info = g.all_datasets[dataset_id]
        ann_infos = g.get_ds_ann_infos(dataset_id)
        for ann_info in ann_infos:
            ann = sly.Annotation.from_json(
                ann_info.annotation, g.project_metas[dataset_info.project_id]
            )
            for label in ann.labels:
                annotators.add(label.geometry.labeler_login)
    return annotators


def get_score(report: List[dict]):
    if "error" in report:
        return "Error"
    for metric in report:
        if metric["metric_name"] == "overall-score":
            if (
                metric["class_gt"] == ""
                and metric["tag_name"] == ""
                and metric["image_gt_id"] == 0
            ):
                return metric["value"]
    return 0


def row_to_str(row):
    return f"{row[0]}. {row[2]}, {row[4]}, {row[6]}"


def get_result_table_data_json(first_rows, second_rows, pairs_scores):
    first_rows_indexes = {row: idx for idx, row in enumerate(first_rows)}
    second_rows_indexes = {row: idx for idx, row in enumerate(second_rows)}
    data = [
        [row_to_str(first_rows[i]), *["" for _ in range(len(second_rows))]]
        for i in range(len(first_rows))
    ]
    for pair, score in pairs_scores.items():
        first_idx = first_rows_indexes[pair[0]]
        second_idx = second_rows_indexes[pair[1]]
        if score != "Error":
            data[first_idx][second_idx + 1] = round(score * 100, 2)
            # data[second_idx][first_idx+1] = round(score*100, 2)
        else:
            data[first_idx][second_idx + 1] = "Error"

    return {"columns": ["", *[row_to_str(row) for row in first_rows]], "data": data}


def select_project(project_id):
    global select_dataset_to_compare
    global select_user_to_compare
    global add_to_compare_btn
    select_dataset_to_compare.loading = True
    select_user_to_compare.loading = True
    add_to_compare_btn.disable()
    datasets = [
        dataset
        for dataset in g.all_datasets.values()
        if dataset.project_id == project_id
    ]
    select_dataset_to_compare.set(
        [
            Select.Item(None, "Combine all datasets"),
            *[Select.Item(dataset.id, dataset.name) for dataset in datasets],
        ]
    )
    select_dataset_to_compare.set_value(None)
    select_user_to_compare.set(
        [
            Select.Item(None, "All users"),
            *[
                Select.Item(login, login)
                for login in get_annotators([dataset.id for dataset in datasets])
            ],
        ]
    )
    select_user_to_compare.set_value(None)
    select_dataset_to_compare.loading = False
    select_user_to_compare.loading = False
    add_to_compare_btn.enable()


@select_project_to_compare.value_changed
def select_project_to_compare_value_changed(value):
    select_project(value)


def select_dataset(dataset_id):
    global add_to_compare_btn
    global select_user_to_compare
    add_to_compare_btn.disable()
    select_user_to_compare.loading = True
    if dataset_id is None:
        selected_project_id = select_project_to_compare.get_value()
        annotators = set(
            annotator
            for annotator in get_annotators(
                [
                    ds.id
                    for ds in g.all_datasets.values()
                    if ds.project_id == selected_project_id
                ]
            )
        )
    else:
        annotators = get_annotators([dataset_id])
    select_user_to_compare.set(
        [
            Select.Item(None, "All users"),
            *[Select.Item(login, login) for login in annotators],
        ]
    )
    select_user_to_compare.set_value(None)
    select_user_to_compare.loading = False
    add_to_compare_btn.enable()


@select_dataset_to_compare.value_changed
def select_datasets_to_compare_value_changed(value):
    select_dataset(value)


def add_user_to_compare(project_name, project_id, dataset_name, dataset_id, user_login):
    rows = compare_table.get_json_data()["raw_rows_data"]
    for row in rows:
        if row == [
            "",
            project_name,
            project_id,
            dataset_name,
            dataset_id,
            user_login,
        ]:
            return
    compare_table.set_data(
        columns=COMPARE_TABLE_COLUMNS,
        rows=[
            *rows,
            [
                "",
                project_name,
                project_id,
                dataset_name,
                dataset_id,
                user_login,
            ],
        ],
        subtitles={c: "" for c in COMPARE_TABLE_COLUMNS},
    )


@add_to_compare_btn.click
def add_to_compare_btn_clicked():
    project_id = select_project_to_compare.get_value()
    dataset_id = select_dataset_to_compare.get_value()
    if dataset_id is None:
        datasets = [ds for ds in g.all_datasets.values() if ds.project_id == project_id]
        if len(datasets) == 0:
            return
        if len(datasets) == 1:
            dataset_id = datasets[0].id
            dataset_name = datasets[0].name
        elif len(datasets) > 1:
            dataset_id = "-"
            dataset_name = ALL_DATASETS
    else:
        dataset_name = g.all_datasets[dataset_id].name
    user_login = select_user_to_compare.get_value()
    users_list = [user_login]
    if user_login is None:
        users_list = [item.value for item in select_user_to_compare.get_items() if item.value is not None]
    if len(users_list) == 0:
        return
    for user_login in users_list:
        add_user_to_compare(
            project_name=g.all_projects[project_id].name,
            project_id=project_id,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            user_login=user_login,
        )


@pop_row_btn.click
def pop_row_btn_clicked():
    selected_row = compare_table.get_selected_row()
    data = compare_table.get_json_data()
    data["raw_rows_data"] = [
        row for row in data["raw_rows_data"] if row != selected_row
    ]
    compare_table.set_data(
        columns=compare_table.columns,
        rows=data["raw_rows_data"],
        subtitles=compare_table.subtitles,
    )


def get_img_infos(project_id, dataset_id):
    if dataset_id == "-":
        datasets_ids = [
            ds.id for ds in g.all_datasets.values() if ds.project_id == project_id
        ]
    else:
        datasets_ids = [dataset_id]
    return [img for ds_id in datasets_ids for img in g.get_ds_img_infos(ds_id)]


def get_ann_infos(project_id, dataset_id):
    if dataset_id == "-":
        datasets_ids = [
            ds.id for ds in g.all_datasets.values() if ds.project_id == project_id
        ]
    else:
        datasets_ids = [dataset_id]
    return [img for ds_id in datasets_ids for img in g.get_ds_ann_infos(ds_id)]

@sly.timeit
def count_common_images(rows_pairs, progress):
    total = 0
    for pair in rows_pairs:
        first_project_id = pair[0][3]
        second_project_id = pair[1][4]
        first_dataset_id = pair[0][5]
        second_dataset_id = pair[1][6]
        first_img_infos = sorted(
            get_img_infos(first_project_id, first_dataset_id), key=lambda x: x.name
        )
        second_img_names = set(img.name for img in get_img_infos(second_project_id, second_dataset_id))
        for first_img in first_img_infos:
            if first_img.name in second_img_names:
                total += 2
                progress.total = progress.total + 2
    return total


@sly.timeit
def get_common_images(pair):
    first_project_id = pair[0][3]
    second_project_id = pair[1][3]
    first_dataset_id = pair[0][5]
    second_dataset_id = pair[1][5]
    first_img_infos = sorted(
        get_img_infos(first_project_id, first_dataset_id), key=lambda x: x.name
    )
    second_img_infos = sorted(
        get_img_infos(second_project_id, second_dataset_id),
        key=lambda x: x.name,
    )
    second_img_name_to_idx = {img.name: i for i, img in enumerate(second_img_infos)}
    paired_infos = []
    for first_img in first_img_infos:
        if first_img.name in second_img_name_to_idx:
            paired_infos.append(
                (first_img, second_img_infos[second_img_name_to_idx[first_img.name]])
            )
    first_img_infos = [paired_info[0] for paired_info in paired_infos]
    second_img_infos = [paired_info[1] for paired_info in paired_infos]
    return first_img_infos, second_img_infos


@sly.timeit
def get_common_ann_infos(pair, first_img_infos, second_img_infos):
    first_project_id = pair[0][3]
    second_project_id = pair[1][3]
    first_dataset_id = pair[0][5]
    second_dataset_id = pair[1][5]
    first_img_id_to_idx = {img.id: i for i, img in enumerate(first_img_infos)}
    second_img_id_to_idx = {img.id: i for i, img in enumerate(second_img_infos)}
    first_imgs_ids = set(img.id for img in first_img_infos)
    second_imgs_ids = set(img.id for img in second_img_infos)
    first_all_ann_infos = get_ann_infos(first_project_id, first_dataset_id)
    second_all_ann_infos = get_ann_infos(second_project_id, second_dataset_id)
    first_ann_infos = sorted(
        [
            ann_info
            for ann_info in first_all_ann_infos
            if ann_info.image_id in first_imgs_ids
        ],
        key=lambda x: first_img_infos[first_img_id_to_idx[x.image_id]].name,
    )
    second_ann_infos = sorted(
        [
            ann_info
            for ann_info in second_all_ann_infos
            if ann_info.image_id in second_imgs_ids
        ],
        key=lambda x: second_img_infos[second_img_id_to_idx[x.image_id]].name,
    )
    return first_ann_infos, second_ann_infos


@sly.timeit
def filter_labels_by_user(first_ann_infos, second_ann_infos, first_meta, second_meta, first_login, second_login):
    for i, first_ann_info in enumerate(first_ann_infos):
        ann = sly.Annotation.from_json(first_ann_info.annotation, first_meta)
        filtered_labels = [
            label
            for label in ann.labels
            if label.geometry.labeler_login == first_login
        ]
        ann = ann.clone(labels=filtered_labels)
        first_ann_infos[i] = AnnotationInfo(
            image_id=first_ann_info.image_id,
            image_name=first_ann_info.image_name,
            annotation=ann.to_json(),
            created_at=first_ann_info.created_at,
            updated_at=first_ann_info.updated_at,
        )
    for i, second_ann_info in enumerate(second_ann_infos):
        ann = sly.Annotation.from_json(second_ann_info.annotation, second_meta)
        filtered_labels = [
            label
            for label in ann.labels
            if label.geometry.labeler_login == second_login
        ]
        ann = ann.clone(labels=filtered_labels)
        second_ann_infos[i] = AnnotationInfo(
            image_id=second_ann_info.image_id,
            image_name=second_ann_info.image_name,
            annotation=ann.to_json(),
            created_at=second_ann_info.created_at,
            updated_at=second_ann_info.updated_at,
        ) 


@sly.timeit
def get_classes(ann_infos, meta):
    class_counts = {}
    for ann_info in ann_infos:
        ann = sly.Annotation.from_json(ann_info.annotation, meta)
        ann_class_counts = ann.stat_class_count(
            [c.name for c in meta.obj_classes]
        )
        ann_class_counts.pop("total")
        for class_name, count in ann_class_counts.items():
            class_counts[class_name] = (
                class_counts.get(class_name, 0) + count
            )
    return set(class_name for class_name, count in class_counts.items() if count > 0)


@sly.timeit
def get_class_matches(first_classes, second_classes):
    return [{"class_gt": class_name, "class_pred": class_name} for class_name in first_classes.intersection(second_classes)]


@sly.timeit
def get_tags_whitelists(first_ann_infos, second_ann_infos, first_meta, second_meta):
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

    tags_whitelist = list(
        first_tag_whitelist.intersection(second_tag_whitelist)
    )
    obj_tags_whitelist = list(
        first_obj_tags_whitelist.intersection(second_obj_tags_whitelist)
    )
    return tags_whitelist, obj_tags_whitelist


@compare_btn.click
def compare_btn_clicked():
    rows = [(i+1, *row) for i, row in enumerate(compare_table.get_json_data()["raw_rows_data"])]
    if len(rows) < 2:
        return

    global result_table
    global report_layout
    global pairs_comparisons_results
    global report_pairs_progress
    global report_calculation_progress
    global repot_preparation_progress
    global name_to_row

    report_layout.hide()
    result_table.loading = True
    rows_pairs = [
        (rows[i], rows[j]) for i in range(len(rows)) for j in range(i+1, len(rows))
    ]
    name_to_row = {row_to_str(row): row for row in rows}
    pair_scores = {}
    report_pairs_progress.show()
    repot_preparation_progress.show()
    report_calculation_progress.show()
    with report_pairs_progress(total=len(rows_pairs)*2, message="Pairs progress") as pairs_progress:
        for i, pair in enumerate(rows_pairs):
            pairs_progress.set_description(f"Pairs progress: {i+1}) {row_to_str(pair[0])} vs {row_to_str(pair[1])}")
            preparation_progress = repot_preparation_progress(total=5, message="Preparing data for the report...")
            if pair not in pairs_comparisons_results:
                # 1. get common images
                first_img_infos, second_img_infos = get_common_images(pair)
                preparation_progress.update(1)

                # 2. get common annotations
                first_ann_infos, second_ann_infos = get_common_ann_infos(pair, first_img_infos, second_img_infos)
                preparation_progress.update(1)

                # 3. filter annotations labels by user
                first_meta = g.project_metas[pair[0][3]]
                second_meta = g.project_metas[pair[1][3]]
                filter_labels_by_user(first_ann_infos, second_ann_infos, first_meta, second_meta, pair[0][6], pair[1][6])
                preparation_progress.update(1)

                # 4. get classes whitelist
                first_classes = get_classes(first_ann_infos, first_meta)
                second_classes = get_classes(second_ann_infos, second_meta)
                class_matches = get_class_matches(first_classes, second_classes)
                preparation_progress.update(1)

                # 5. get tags whitelists
                tags_whitelist, obj_tags_whitelist = get_tags_whitelists(first_ann_infos, second_ann_infos, first_meta, second_meta)
                preparation_progress.update(1)
                
                
                calculation_progress = report_calculation_progress(total=len(first_img_infos), message=f'Calculating consensus report for pair: "{row_to_str(pair[0])}" and "{row_to_str(pair[1])}"')
                report, difference_geometries = calculate_exam_report(
                    united_meta=g.project_metas[pair[0][3]],
                    img_infos_gt=first_img_infos,
                    img_infos_pred=second_img_infos,
                    ann_infos_gt=first_ann_infos,
                    ann_infos_pred=second_ann_infos,
                    class_matches=class_matches,
                    tags_whitelist=tags_whitelist,
                    obj_tags_whitelist=obj_tags_whitelist,
                    iou_threshold=0.5,
                    progress=calculation_progress,
                )
                pairs_comparisons_results[
                    (pair[0], pair[1])
                ] = ComparisonResult(
                    pair=pair,
                    first_meta=first_meta,
                    second_meta=second_meta,
                    first_images=first_img_infos,
                    second_images=second_img_infos,
                    first_annotations_jsons=[ai.annotation for ai in first_ann_infos],
                    second_annotations_jsons=[ai.annotation for ai in second_ann_infos],
                    tags=list(set(tags_whitelist) | set(obj_tags_whitelist)),
                    classes=[cm["class_gt"] for cm in class_matches],
                    first_classes=list(first_classes),
                    second_classes=list(second_classes),
                    report=report,
                    differences=difference_geometries,
                )
                report_calculation_progress.show()
                score = get_score(report)
                pair_scores[pair] = score
                pairs_progress.update(1)

                calculation_progress = report_calculation_progress(total=len(first_img_infos), message=f'Calculating consensus report for pair: "{row_to_str(pair[1])}" and "{row_to_str(pair[0])}"')
                report, difference_geometries = calculate_exam_report(
                    united_meta=g.project_metas[pair[1][3]],
                    img_infos_gt=second_img_infos,
                    img_infos_pred=first_img_infos,
                    ann_infos_gt=second_ann_infos,
                    ann_infos_pred=first_ann_infos,
                    class_matches=class_matches,
                    tags_whitelist=tags_whitelist,
                    obj_tags_whitelist=obj_tags_whitelist,
                    iou_threshold=0.5,
                    progress=calculation_progress,
                )
                pairs_comparisons_results[
                    (tuple(pair[::-1]))
                ] = ComparisonResult(
                    pair=tuple(pair[::-1]),
                    first_meta=second_meta,
                    second_meta=first_meta,
                    first_images=second_img_infos,
                    second_images=first_img_infos,
                    first_annotations_jsons=[ai.annotation for ai in second_ann_infos],
                    second_annotations_jsons=[ai.annotation for ai in first_ann_infos],
                    tags=list(set(tags_whitelist) | set(obj_tags_whitelist)),
                    classes=[cm["class_gt"] for cm in class_matches],
                    first_classes=list(second_classes),
                    second_classes=list(first_classes),
                    report=report,
                    differences=difference_geometries,
                )
                report_calculation_progress.show()
                score = get_score(report)
                pair_scores[tuple(pair[::-1])] = score
                pairs_progress.update(1)
            else:
                report = pairs_comparisons_results[
                    pair
                ].get_report()
                score = get_score(report)
                pair_scores[pair] = score
                pairs_progress.update(1)
                report = pairs_comparisons_results[
                    tuple(pair[::-1])
                ].get_report()
                score = get_score(report)
                pair_scores[tuple(pair[::-1])] = score
                pairs_progress.update(1)

    result_table.read_json(
        get_result_table_data_json([tuple(rows[i]) for i in range(len(rows))], [tuple(rows[i]) for i in range(len(rows))], pair_scores)
    )
    report_pairs_progress.hide()
    repot_preparation_progress.hide()
    report_calculation_progress.hide()
    result_table.loading = False
    result_table.show()
    consensus_report_notification.show()


@result_table.click
def result_table_clicked(datapoint):
    global name_to_row
    row_name = datapoint.row[""]
    column_name = datapoint.column_name
    
    if column_name == "" or row_name == column_name:
        return

    global consensus_report_error_notification
    global report_layout
    global pairs_comparisons_results
    global consensus_report_text
    global consensus_report_notification
    global consensus_report_details

    pair = (name_to_row[row_name], name_to_row[column_name])
    comparison_result = pairs_comparisons_results[pair]
    comparison_result: ComparisonResult
    if comparison_result.error_message is not None:
        consensus_report_error_notification.show()
        consensus_report_error_notification.set(title="Error", description=f'Error occured while calculating consensus report. Error Message: "{comparison_result.error_message}"')
        return
    consensus_report_error_notification.hide()
    report_layout.loading = True
    consensus_report_details.loading = True
    report_layout.show()
    render_report(
        report=comparison_result.get_report(),
        gt_imgs=comparison_result.first_images,
        pred_imgs=comparison_result.second_images,
        gt_anns=comparison_result.get_first_annotations(),
        pred_anns=comparison_result.get_second_annotations(),
        diffs=comparison_result.get_differences(),
        classes=comparison_result.classes,
        tags=comparison_result.tags,
        first_name=row_name,
        second_name=column_name,
    )
    consensus_report_notification.hide()
    consensus_report_text.set("<h1>Consensus report</h1>", status="text")
    
    selected_pair_first.set(
        project_name=comparison_result.pair[0][2],
        dataset_name=comparison_result.pair[0][4],
        user_login=comparison_result.pair[0][6],
        classes=[(cls.name, cls.color) for cls in comparison_result.first_meta.obj_classes if cls.name in comparison_result.first_classes]
    )
    selected_pair_second.set(
        project_name=comparison_result.pair[1][2],
        dataset_name=comparison_result.pair[1][4],
        user_login=comparison_result.pair[1][6],
        classes=[(cls.name, cls.color) for cls in comparison_result.second_meta.obj_classes if cls.name in comparison_result.second_classes]
    )
    consensus_report_classes.set(text="".join(wrap_in_tag(cls.name, cls.color) for cls in comparison_result.first_meta.obj_classes if cls.name in comparison_result.classes), status="text")
    
    consensus_report_text.show()
    consensus_report_details.show()
    consensus_report_details.loading = False
    report_layout.loading = False


if g.project_id:
    select_project_to_compare_field = Field(
        title="Project", content=ProjectThumbnail(g.all_projects[g.project_id])
    )
    select_project_to_compare.set_value(g.project_id)
    select_project(g.project_id)
    if g.dataset_id:
        select_dataset_to_compare.set_value(g.dataset_id)
        select_dataset_to_compare_field = Field(
            title="Dataset",
            content=DatasetThumbnail(
                g.all_projects[g.project_id],
                g.all_datasets[g.dataset_id],
                show_project_name=False,
            ),
        )
        select_dataset(g.dataset_id)


layout = Container(
    widgets=[
        Container(
            widgets=[
                Card(
                    title="1️⃣ Select Users to compare",
                    description="Select datasets and users to compare and click '+ ADD' button",
                    content=Container(
                        widgets=[
                            workspace_thumbnail,
                            select_project_to_compare_field,
                            select_dataset_to_compare_field,
                            Field(title="User", content=select_user_to_compare),
                            add_to_compare_btn,
                        ]
                    ),
                ),
                Card(
                    title="2️⃣ Selected Users",
                    description="Here you can see a list of selected users. You can remove a user from the list by selecting it and clicking 'REMOVE' button",
                    content=Container(widgets=[compare_table, pop_row_btn]),
                ),
            ],
            direction="horizontal",
            overflow="wrap",
        ),
        Card(
            title="3️⃣ Compare Results",
            description="Click on 'CALCULATE CONSENSUS' button to see comparison matrix. Value in a table cell is a consensus score between two users",
            content=Container(widgets=[report_pairs_progress, repot_preparation_progress, report_calculation_progress, result_table]),
            content_top_right=compare_btn,
        ),
        consensus_report_notification,
        consensus_report_error_notification,
        consensus_report_text,
        consensus_report_details,
        report_layout,
    ]
)
