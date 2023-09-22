from typing import Dict, List, Tuple
from collections import namedtuple
import json
from pathlib import Path
import traceback
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
    InputNumber,
    Checkbox,
    OneOf,
    SelectWorkspace,
    Input,
    SelectTagMeta,
    InputTag,
    Editor,
)

import src.globals as g
import src.utils as utils
from src.ui.report import (
    render_report,
    layout as report_layout,
    report_to_dict,
)
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


Row = namedtuple(
    typename="row",
    field_names=[
        "index",
        "project_name",
        "project_id",
        "dataset_name",
        "dataset_id",
        "annotator_login",
    ],
)


class ComparisonResult:
    @sly.timeit
    def __init__(
        self,
        pair: Tuple[Row, Row],
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
        first_name = ", ".join(str(p) for p in pair[0][2:])
        second_name = ", ".join(str(p) for p in pair[1][2:])
        self.DIR_PATH = f"{g.TEMP_DATA_PATH}/compare_results/{first_name}/{second_name}"
        self.pair = pair
        self.first_meta = first_meta
        self.second_meta = second_meta
        self.first_images = first_images
        self.second_images = second_images
        self._first_annotations_path = self.save_anns_jsons("anns_first", first_annotations_jsons)
        self._second_annotations_path = self.save_anns_jsons(
            "anns_second", second_annotations_jsons
        )
        self.tags = tags
        self.classes = classes
        self.first_classes = first_classes
        self.second_classes = second_classes
        self._report_path = self.save_report(report)
        self._differences_path = self.save_differences(differences)
        try:
            self.error_message = report["error"]
        except (KeyError, TypeError):
            self.error_message = None

    def mkdir(self):
        path = Path(self.DIR_PATH)
        if not path.exists():
            path.mkdir(parents=True)
        return path

    @sly.timeit
    def save_differences(self, difference_geometries: List[sly.Bitmap]):
        dir_path = self.mkdir()
        filepath = dir_path.joinpath(f"diffs.json")
        with open(filepath, "w") as f:
            json.dump(
                [
                    None if geometry is None else geometry.to_json()
                    for geometry in difference_geometries
                ],
                f,
            )
        return filepath

    @sly.timeit
    def save_anns_jsons(self, filename, anns: List[Dict]):
        dir_path = self.mkdir()
        filepath = dir_path.joinpath(f"{filename}.json")
        with open(filepath, "w") as f:
            json.dump(anns, f)
        return filepath

    @sly.timeit
    def save_report(self, report):
        dir_path = self.mkdir()
        filepath = dir_path.joinpath(f"report.json")
        with open(filepath, "w") as f:
            json.dump(report, f)
        return filepath

    def get_first_annotations(self):
        with open(self._first_annotations_path, "r") as f:
            return [
                sly.Annotation.from_json(ann_json, self.first_meta) for ann_json in json.load(f)
            ]

    def get_second_annotations(self):
        with open(self._second_annotations_path, "r") as f:
            return [
                sly.Annotation.from_json(ann_json, self.second_meta) for ann_json in json.load(f)
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
        self._classes_text.set(
            "".join([utils.wrap_in_tag(class_name, color) for class_name, color in self._classes]),
            status="text",
        )

    def set(self, project_name, dataset_name, user_login, classes):
        self._project_name = project_name
        self._dataset_name = dataset_name
        self._user_login = user_login
        self._classes = classes
        self.update()


# Widgets
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
select_project_to_compare_field = Field(title="Project", content=select_project_to_compare)
select_dataset_to_compare = Select(items=[Select.Item(None, "Combine all datasets")])
select_dataset_to_compare_field = Field(title="Dataset", content=select_dataset_to_compare)
select_user_to_compare = Select(items=[Select.Item(None, "All users")])
add_to_compare_btn = Button("add", icon="zmdi zmdi-plus", button_size="small")
compare_table = RadioTable(columns=COMPARE_TABLE_COLUMNS, rows=[])
pop_row_btn = Button("remove", button_size="small")
compare_btn = Button("calculate consensus")
threshold_input = InputNumber(value=0.5, min=0, max=1, controls=False)
segmentation_mode_checkbox = Checkbox("Enable")
report_progress_current_pair_first = Text()
report_progress_current_pair_second = Text()
report_progress_current_pair = Flexbox(
    widgets=[
        Text("Current pair:"),
        report_progress_current_pair_first,
        Text("vs"),
        report_progress_current_pair_second,
    ]
)
report_progress_current_pair.hide()
report_calculation_progress = Progress(
    "Calculating consensus report for pair: ", show_percents=False, hide_on_finish=False
)
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
            Flexbox(widgets=[Text("<h3>Second:</h3>"), selected_pair_second.layout], gap=20),
            Flexbox(
                widgets=[Text("<b>Report Classes:</b>"), consensus_report_classes],
                gap=15,
            ),
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
actions_select_which_images = Select(
    items=[Select.Item("below", "Below threshold"), Select.Item("above", "Above threshold")]
)
actions_select_threshold = InputNumber(value=50, min=0, max=100, controls=False)
actions_select_metric = Select(
    items=[
        Select.Item("overall-score", "Overall Score"),
        Select.Item("matches-f1", "Objects Score"),
        Select.Item("tags-f1", "Tags Score"),
        Select.Item("iou", "Geometry Score"),
    ]
)
# actions_save_inputs_dst = DestinationProject(None, project_type=sly.ProjectType.IMAGES)
actions_save_inputs_ws = SelectWorkspace(team_id=g.TEAM_ID)
actions_save_inputs_pr_name = Input()
actions_save_inputs_ds_name = Input()
actions_save_inputs = Container(
    widgets=[
        Text("<p>Copy images to another project</p>"),
        Field(title="Workspace", description="Select workspace", content=actions_save_inputs_ws),
        Field(
            title="Project Name",
            description="Input project name. If project does not exist, new project will be created.",
            content=actions_save_inputs_pr_name,
        ),
        Field(
            title="Dataset Name",
            description="Input dataset name",
            content=actions_save_inputs_ds_name,
        ),
    ]
)
actions_tag_inputs_tag_meta = SelectTagMeta(project_meta=sly.ProjectMeta())
actions_tag_inputs_tag_value = InputTag(tag_meta=sly.TagMeta("", value_type=sly.TagValueType.NONE))
actions_tag_inputs = Container(
    widgets=[
        Text("<p>Add or remove tag from images</p>"),
        Field(title="Select Tag", content=actions_tag_inputs_tag_meta),
        Field(
            title="Select Tag Value",
            description="Disable tag to delete it",
            content=actions_tag_inputs_tag_value,
        ),
    ]
)
actions_lj_inputs_name = Input()
actions_lj_inputs_user_ids = Select(
    items=[Select.Item(user.id, user.login) for user in g.all_users.values()],
    multiple=True,
    filterable=True,
)
actions_lj_inputs_readme = Editor(language_mode="plain_text")
actions_lj_inputs_description = Editor(language_mode="plain_text")
actions_lj_inputs_classes_to_label = Select(items=[], multiple=True, filterable=True)
actions_lj_inputs_tags_to_label = Select(items=[], multiple=True, filterable=True)
actions_labeling_job_inputs = Container(
    widgets=[
        Text("<p>Create labeling job for images</p>"),
        Field(title="Name", content=actions_lj_inputs_name),
        Field(
            title="Users",
            description="Select at least 1 user. Labeling job will be created for each user. If images are from different datasets, a labeling job will created for each dataset.",
            content=actions_lj_inputs_user_ids,
        ),
        Field(title="Description", content=actions_lj_inputs_description),
        Field(title="Readme", content=actions_lj_inputs_readme),
        Field(title="Classes to label", content=actions_lj_inputs_classes_to_label),
        Field(title="Tags to label", content=actions_lj_inputs_tags_to_label),
    ]
)
actions_select_action = Select(
    items=[
        Select.Item("save", "Copy to project", actions_save_inputs),
        Select.Item("assign_tag", "Assign Tag", actions_tag_inputs),
        Select.Item("labeling_job", "Create Labeling Job", actions_labeling_job_inputs),
    ]
)
actions_action_settings = OneOf(actions_select_action)
actions_run_btn = Button("Run", icon="zmdi zmdi-play", button_size="small")
actions_progress = Progress()
actions_progress.hide()
actions_total = Text(text="", status="success")
actions_total.hide()
actions_card = Card(
    title="5️⃣ Actions",
    description="Perform different actions with images",
    content=Container(
        widgets=[
            Field(title="Action", content=actions_select_action),
            actions_action_settings,
            Text("<h3>Select images for action</h3>"),
            Field(
                title="Score Threshold in %",
                description="Input threshold of metric score",
                content=actions_select_threshold,
            ),
            Field(title="Score Metric", description="Choose metric", content=actions_select_metric),
            Field(
                title="Condition",
                description="Select condition for images selection",
                content=actions_select_which_images,
            ),
            actions_run_btn,
            actions_progress,
            actions_total,
        ]
    ),
    collapsable=True,
)
actions_card.collapse()


# global variables
pairs_comparisons_results = {}
name_to_row = {}


def get_save_settings():
    ws_id = actions_save_inputs_ws.get_selected_id()
    pr_name = actions_save_inputs_pr_name.get_value()
    ds_name = actions_save_inputs_ds_name.get_value()
    return ws_id, pr_name, ds_name


def get_images_for_actions(metric, passmark, result):
    global name_to_row
    cell_data = result_table.get_selected_cell(sly.app.StateJson())
    row_name = cell_data["row"][""]
    column_name = cell_data["column_name"]
    if column_name == "" or row_name == column_name:
        return []
    pair = (name_to_row[row_name], name_to_row[column_name])
    comparison_result = pairs_comparisons_results[pair]
    comparison_result: ComparisonResult
    if comparison_result.error_message is not None:
        return []
    if len(comparison_result.first_images) != len(comparison_result.second_images):
        return []
    report = comparison_result.get_report()
    report_dict = report_to_dict(report)
    if result == "above":
        comparator = lambda x: x >= passmark
    else:
        comparator = lambda x: x < passmark
    res = []
    for gt_img, pred_img, ann in zip(
        comparison_result.first_images,
        comparison_result.second_images,
        comparison_result.get_second_annotations(),
    ):
        try:
            metric_value = report_dict[metric][gt_img.id][("", "")]
        except KeyError:
            metric_value = 0
        if comparator(metric_value):
            res.append((gt_img, pred_img, ann))
    return res


def actions_save_func(
    ws_id, pr_name, ds_name, img_infos: List[sly.ImageInfo], anns: List[sly.Annotation]
):
    if len(img_infos) != len(anns):
        raise RuntimeError("Number of images and annotations is not equal")
    pr_info = utils.get_project_by_name(ws_id, pr_name)
    if pr_info is None:
        pr_info = g.api.project.create(ws_id, pr_name, change_name_if_conflict=True)
    ds_info = utils.get_dataset_by_name(pr_info.id, ds_name)
    if ds_info is None:
        ds_info = g.api.dataset.create(pr_info.id, ds_name, change_name_if_conflict=True)
    src_ds_ids = set(img_info.dataset_id for img_info in img_infos)
    src_datasets = set(utils.get_dataset_by_id(ds_id) for ds_id in src_ds_ids)
    src_projects = set(utils.get_project_by_id(ds.project_id) for ds in src_datasets)
    for src_pr in src_projects:
        g.api.project.merge_metas(src_pr.id, pr_info.id)
    new_img_infos = g.api.image.upload_ids(
        dataset_id=ds_info.id,
        names=[img_info.name for img_info in img_infos],
        ids=[img_info.id for img_info in img_infos],
        infos=img_infos,
    )
    g.api.annotation.upload_anns(img_ids=[img_info.id for img_info in img_infos], anns=anns)
    return new_img_infos


def get_tag_settings():
    tag_meta = actions_tag_inputs_tag_meta.get_selected_item()
    tag = actions_tag_inputs_tag_value.get_tag()
    return tag_meta, tag


def actions_assign_tag_func(
    tag: sly.Tag, img_infos: List[sly.ImageInfo], anns: List[sly.Annotation]
):
    if len(img_infos) != len(anns):
        raise RuntimeError("Number of images and annotations is not equal")
    updated_anns = []
    for ann in anns:
        existing_tag = ann.img_tags.get(tag.name)
        existing_tag: sly.Tag
        if existing_tag is None:
            if tag is not None:
                ann = ann.add_tag(tag)
        elif tag is None:
            ann = ann.delete_tag(existing_tag)
        elif existing_tag.value != tag.value:
            ann = ann.delete_tag(existing_tag).add_tag(tag)
        updated_anns.append(ann)
    g.api.annotation.upload_anns([img_info.id for img_info in img_infos], updated_anns)
    return img_infos, ann


def get_lj_settings():
    labeling_job_name = actions_lj_inputs_name.get_value()
    if labeling_job_name is None or labeling_job_name.isspace():
        raise RuntimeError("Labeling job name is empty")
    user_ids = actions_lj_inputs_user_ids.get_value()
    if not user_ids:
        raise RuntimeError("No users selected")
    readme = actions_lj_inputs_readme.get_value()
    description = actions_lj_inputs_description.get_value()
    classes_to_label = actions_lj_inputs_classes_to_label.get_value()
    if not classes_to_label:
        classes_to_label = []
    tags_to_label = actions_lj_inputs_tags_to_label.get_value()
    if not tags_to_label:
        tags_to_label = []
    return labeling_job_name, user_ids, readme, description, classes_to_label, tags_to_label


def actions_lj_func(
    labeling_job_name,
    user_ids,
    readme,
    description,
    classes_to_label,
    tags_to_label,
    img_infos: List[sly.ImageInfo],
    progress: Progress,
):
    ds_ids = set()
    ds_img_ids = {}
    for img_info in img_infos:
        ds_ids.add(img_info.dataset_id)
        ds_img_ids.setdefault(img_info.dataset_id, []).append(img_info.id)
    labeling_jobs = []
    with progress(total=len(ds_ids) * len(user_ids), message="Creating labeling jobs...") as pbar:
        for ds_id in ds_ids:
            labeling_jobs.extend(
                g.api.labeling_job.create(
                    name=labeling_job_name,
                    dataset_id=ds_id,
                    user_ids=user_ids,
                    readme=readme,
                    description=description,
                    classes_to_label=classes_to_label,
                    tags_to_label=tags_to_label,
                    images_ids=ds_img_ids[ds_id],
                )
            )
            pbar.update(len(user_ids))
    return labeling_jobs


@actions_run_btn.click
def actions_run():
    try:
        actions_total.hide()
        metric = actions_select_metric.get_value()
        passmark = actions_select_threshold.get_value() / 100
        result = actions_select_which_images.get_value()
        action = actions_select_action.get_value()
        items = get_images_for_actions(metric, passmark, result)
        if action == "save":
            actions_progress.show()
            ws_id, pr_name, ds_name = get_save_settings()
            with actions_progress(
                iterable=items, message=f'Saving images to project "{pr_name}"...'
            ) as pbar:
                for batch in sly.batched(items):
                    pred_imgs = [pred_img for _, pred_img, _ in batch]
                    anns = [ann for _, _, ann in batch]
                    _ = actions_save_func(ws_id, pr_name, ds_name, pred_imgs, anns)
                    pbar.update(len(batch))
            actions_total.text = f'Saved {len(items)} images to project "{pr_name}"'
            actions_total.show()
        elif action == "assign_tag":
            actions_progress.show()
            tag_meta, tag = get_tag_settings()
            action_name = (
                f'Removing tag "{tag_meta.name}" from images...'
                if tag is None
                else f'Updating tag "{tag_meta.name}" in images...'
            )
            with actions_progress(iterable=items, message=action_name) as pbar:
                for batch in sly.batched(items):
                    pred_imgs = [pred_img for _, pred_img, _ in batch]
                    anns = [ann for _, _, ann in batch]
                    _, _ = actions_assign_tag_func(tag, pred_imgs, anns)
                    pbar.update(len(batch))
            action_name = "removed from" if tag is None else "assigned to"
            actions_total.text = f"Tag {action_name} {len(items)} images"
            actions_total.show()
        elif action == "labeling_job":
            (
                labeling_job_name,
                user_ids,
                readme,
                description,
                classes_to_label,
                tags_to_label,
            ) = get_lj_settings()
            created_labeling_jobgs = actions_lj_func(
                labeling_job_name=labeling_job_name,
                user_ids=user_ids,
                readme=readme,
                description=description,
                classes_to_label=classes_to_label,
                tags_to_label=tags_to_label,
                img_infos=[img_info for img_info, _, _ in items],
                progress=actions_progress,
            )
            actions_total.text = (
                f"Created {len(created_labeling_jobgs)} labeling jobs for {len(user_ids)} users"
            )
            actions_total.show()
        else:
            raise RuntimeError("Unknown action")
    except Exception:
        sly.logger.error("Error occured while performing action", exc_info=traceback.format_exc())
        sly.app.show_dialog("Error", "Error occured while performing action")
    finally:
        actions_progress.hide()


@actions_tag_inputs_tag_meta.value_changed
def tag_meta_changed(tag_meta):
    if tag_meta is None:
        return
    actions_tag_inputs_tag_value.set_tag_meta(tag_meta)


def row_to_str(row: Row):
    return f"{row.index}. {row.project_name}, {row.dataset_name}, {row.annotator_login}"


def get_result_table_data_json(first_rows: List[Row], second_rows: List[Row], pairs_scores: Dict):
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
        else:
            data[first_idx][second_idx + 1] = "Error"

    return {"columns": ["", *[row_to_str(row) for row in first_rows]], "data": data}


def set_actions(project_meta: sly.ProjectMeta):
    actions_tag_inputs.loading = True
    actions_tag_inputs_tag_meta.set_project_meta(project_meta)
    obj_classes = list(project_meta.obj_classes)
    if obj_classes:
        actions_tag_inputs_tag_meta.set_name(obj_classes[0].name)
    actions_tag_inputs.loading = False
    actions_labeling_job_inputs.loading = True
    actions_lj_inputs_classes_to_label.set(
        items=[
            Select.Item(obj_class.name, obj_class.name) for obj_class in project_meta.obj_classes
        ]
    )
    actions_lj_inputs_classes_to_label.set_value([])
    actions_lj_inputs_tags_to_label.set(
        items=[Select.Item(tag_meta.name, tag_meta.name) for tag_meta in project_meta.tag_metas]
    )
    actions_lj_inputs_tags_to_label.set_value([])
    actions_labeling_job_inputs.loading = False


def select_project(project_id):
    global select_dataset_to_compare
    global select_user_to_compare
    global add_to_compare_btn
    select_dataset_to_compare.loading = True
    select_user_to_compare.loading = True
    add_to_compare_btn.disable()
    datasets = [dataset for dataset in g.all_datasets.values() if dataset.project_id == project_id]
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
                for login in utils.get_annotators([dataset.id for dataset in datasets])
            ],
        ]
    )
    select_user_to_compare.set_value(None)
    select_dataset_to_compare.loading = False
    select_user_to_compare.loading = False
    add_to_compare_btn.enable()


select_project_to_compare.value_changed(select_project)


def select_dataset(dataset_id):
    global add_to_compare_btn
    global select_user_to_compare
    add_to_compare_btn.disable()
    select_user_to_compare.loading = True
    if dataset_id is None:
        selected_project_id = select_project_to_compare.get_value()
        annotators = set(
            annotator
            for annotator in utils.get_annotators(
                [ds.id for ds in g.all_datasets.values() if ds.project_id == selected_project_id]
            )
        )
    else:
        annotators = utils.get_annotators([dataset_id])
    select_user_to_compare.set(
        [
            Select.Item(None, "All users"),
            *[Select.Item(login, login) for login in annotators],
        ]
    )
    select_user_to_compare.set_value(None)
    select_user_to_compare.loading = False
    add_to_compare_btn.enable()


select_dataset_to_compare.value_changed(select_dataset)


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
        users_list = [
            item.value for item in select_user_to_compare.get_items() if item.value is not None
        ]
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
    data["raw_rows_data"] = [row for row in data["raw_rows_data"] if row != selected_row]
    compare_table.set_data(
        columns=compare_table.columns,
        rows=data["raw_rows_data"],
        subtitles=compare_table.subtitles,
    )


@compare_btn.click
def compare_btn_clicked():
    global compare_table

    rows = [
        Row(i + 1, *r[1:]) for i, r in enumerate(compare_table.get_json_data()["raw_rows_data"])
    ]
    if len(rows) < 2:
        return

    global result_table
    global report_layout
    global report_progress_current_pair_first
    global report_progress_current_pair_second
    global report_progress_current_pair
    global report_calculation_progress
    global pairs_comparisons_results
    global name_to_row

    result_table.hide()
    report_layout.hide()
    report_progress_current_pair.show()
    report_calculation_progress.show()

    name_to_row = {row_to_str(row): row for row in rows}
    pairs_comparisons_results = {}
    utils.ds_img_infos = {}
    utils.ds_ann_infos = {}

    rows_pairs = [(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]
    threshold = threshold_input.get_value()
    segmentation_mode = segmentation_mode_checkbox.is_checked()
    pair_scores = {}

    for first, second in rows_pairs:
        if (first, second) not in pairs_comparisons_results:
            report_progress_current_pair_first.text = row_to_str(first)
            report_progress_current_pair_second.text = row_to_str(second)
            with report_calculation_progress(
                message="Preparing data for the report...", total=100
            ) as pbar:
                # 0. get data
                first_meta = g.project_metas[first.project_id]
                second_meta = g.project_metas[second.project_id]
                first_img_infos = utils.get_img_infos(first.project_id, first.dataset_id)
                pbar.update(10)
                first_ann_infos = utils.get_ann_infos(first.project_id, first.dataset_id)
                pbar.update(10)
                second_img_infos = utils.get_img_infos(second.project_id, second.dataset_id)
                pbar.update(10)
                second_ann_infos = utils.get_ann_infos(second.project_id, second.dataset_id)
                pbar.update(10)

                # 1. get common images
                first_img_infos, second_img_infos = utils.get_common_images(
                    first_img_infos, second_img_infos
                )
                pbar.update(10)

                # 2. get common annotations
                first_ann_infos, second_ann_infos = utils.get_common_ann_infos(
                    first_img_infos, second_img_infos, first_ann_infos, second_ann_infos
                )
                pbar.update(10)

                # 3. filter annotations labels by user
                utils.filter_labels_by_user(
                    first_ann_infos,
                    second_ann_infos,
                    first_meta,
                    second_meta,
                    first.annotator_login,
                    second.annotator_login,
                )
                pbar.update(10)

                # 4. get classes whitelist
                first_classes = utils.get_classes(first_ann_infos, first_meta)
                second_classes = utils.get_classes(second_ann_infos, second_meta)
                class_matches = utils.get_class_matches(first_classes, second_classes)
                pbar.update(10)

                # 5. get tags whitelists
                tags_whitelist, obj_tags_whitelist = utils.get_tags_whitelists(
                    first_ann_infos, second_ann_infos, first_meta, second_meta
                )
                pbar.update(10)

            with report_calculation_progress(
                total=len(first_img_infos),
                message="Calculating consensus report...",
            ) as pbar:
                report, difference_geometries = calculate_exam_report(
                    united_meta=g.project_metas[first.project_id],
                    img_infos_gt=first_img_infos,
                    img_infos_pred=second_img_infos,
                    ann_infos_gt=first_ann_infos,
                    ann_infos_pred=second_ann_infos,
                    class_mapping=class_matches,
                    tags_whitelist=tags_whitelist,
                    obj_tags_whitelist=obj_tags_whitelist,
                    iou_threshold=threshold,
                    progress=pbar,
                    segmentation_mode=segmentation_mode,
                )
            with report_calculation_progress(
                total=len(first_img_infos),
                message="Saving consensus report...",
            ) as pbar:
                pairs_comparisons_results[(first, second)] = ComparisonResult(
                    pair=(first, second),
                    first_meta=first_meta,
                    second_meta=second_meta,
                    first_images=first_img_infos,
                    second_images=second_img_infos,
                    first_annotations_jsons=[ai.annotation for ai in first_ann_infos],
                    second_annotations_jsons=[ai.annotation for ai in second_ann_infos],
                    tags=list(set(tags_whitelist) | set(obj_tags_whitelist)),
                    classes=list(class_matches.keys()),
                    first_classes=list(first_classes),
                    second_classes=list(second_classes),
                    report=report,
                    differences=difference_geometries,
                )
            report_calculation_progress.show()
            score = utils.get_score(report)
            pair_scores[(first, second)] = score

            report_progress_current_pair_first.text = row_to_str(second)
            report_progress_current_pair_second.text = row_to_str(first)
            with report_calculation_progress(
                total=len(first_img_infos),
                message="Calculating consensus report...",
            ) as pbar:
                report, difference_geometries = calculate_exam_report(
                    united_meta=g.project_metas[second.project_id],
                    img_infos_gt=second_img_infos,
                    img_infos_pred=first_img_infos,
                    ann_infos_gt=second_ann_infos,
                    ann_infos_pred=first_ann_infos,
                    class_mapping=class_matches,
                    tags_whitelist=tags_whitelist,
                    obj_tags_whitelist=obj_tags_whitelist,
                    iou_threshold=threshold,
                    progress=pbar,
                    segmentation_mode=segmentation_mode,
                )
            with report_calculation_progress(
                total=len(first_img_infos),
                message="Saving consensus report...",
            ) as pbar:
                pairs_comparisons_results[(second, first)] = ComparisonResult(
                    pair=(second, first),
                    first_meta=second_meta,
                    second_meta=first_meta,
                    first_images=second_img_infos,
                    second_images=first_img_infos,
                    first_annotations_jsons=[ai.annotation for ai in second_ann_infos],
                    second_annotations_jsons=[ai.annotation for ai in first_ann_infos],
                    tags=list(set(tags_whitelist) | set(obj_tags_whitelist)),
                    classes=list(class_matches.keys()),
                    first_classes=list(second_classes),
                    second_classes=list(first_classes),
                    report=report,
                    differences=difference_geometries,
                )
            report_calculation_progress.show()
            score = utils.get_score(report)
            pair_scores[tuple((first, second)[::-1])] = score
        else:
            report_progress_current_pair_first.text = row_to_str(first)
            report_progress_current_pair_second.text = row_to_str(second)
            report = pairs_comparisons_results[(first, second)].get_report()
            score = utils.get_score(report)
            pair_scores[(first, second)] = score

            report_progress_current_pair_first.text = row_to_str(second)
            report_progress_current_pair_second.text = row_to_str(first)
            report = pairs_comparisons_results[(second, first)].get_report()
            score = utils.get_score(report)
            pair_scores[(second, first)] = score

    result_table.read_json(get_result_table_data_json(rows, rows, pair_scores))
    report_progress_current_pair.hide()
    report_calculation_progress.hide()
    result_table.show()
    consensus_report_notification.show()


@result_table.click
@sly.timeit
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

    consensus_report_notification.hide()
    pair = (name_to_row[row_name], name_to_row[column_name])
    comparison_result = pairs_comparisons_results[pair]
    comparison_result: ComparisonResult
    if comparison_result.error_message is not None:
        consensus_report_error_notification.show()
        consensus_report_error_notification.set(
            title="Error",
            description=f'Error occured while calculating consensus report. Error Message: "{comparison_result.error_message}"',
        )
        return
    consensus_report_error_notification.hide()
    report_layout.loading = True
    consensus_report_details.loading = True
    report_layout.show()
    consensus_report_details.show()

    selected_pair_first.set(
        project_name=comparison_result.pair[0].project_name,
        dataset_name=comparison_result.pair[0].dataset_name,
        user_login=comparison_result.pair[0].annotator_login,
        classes=[
            (cls.name, cls.color)
            for cls in comparison_result.first_meta.obj_classes
            if cls.name in comparison_result.first_classes
        ],
    )
    selected_pair_second.set(
        project_name=comparison_result.pair[1].project_name,
        dataset_name=comparison_result.pair[1].dataset_name,
        user_login=comparison_result.pair[1].annotator_login,
        classes=[
            (cls.name, cls.color)
            for cls in comparison_result.second_meta.obj_classes
            if cls.name in comparison_result.second_classes
        ],
    )
    consensus_report_classes.set(
        text="".join(
            utils.wrap_in_tag(cls.name, cls.color)
            for cls in comparison_result.first_meta.obj_classes
            if cls.name in comparison_result.classes
        ),
        status="text",
    )

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

    set_actions(comparison_result.second_meta)

    consensus_report_text.show()
    consensus_report_details.show()
    consensus_report_details.loading = False
    report_layout.loading = False


if g.PROJECT_ID:
    select_project_to_compare_field = Field(
        title="Project", content=ProjectThumbnail(g.all_projects[g.PROJECT_ID])
    )
    select_project_to_compare.set_value(g.PROJECT_ID)
    select_project(g.PROJECT_ID)
    if g.DATASET_ID:
        select_dataset_to_compare.set_value(g.DATASET_ID)
        select_dataset_to_compare_field = Field(
            title="Dataset",
            content=DatasetThumbnail(
                g.all_projects[g.PROJECT_ID],
                g.all_datasets[g.DATASET_ID],
                show_project_name=False,
            ),
        )
        select_dataset(g.DATASET_ID)
else:
    if len(select_project_to_compare_items) > 0:
        select_project(select_project_to_compare_items[0].value)
    select_dataset(select_dataset_to_compare.get_items()[0].value)


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
            title="3️⃣ Parameters",
            description="Select parameters for report calculation",
            content=Container(
                widgets=[
                    Field(
                        title="Segmentation mode",
                        description='If enabled, geometries of type "Bitmap" and "Polygon" will be treated as segmentation. Label that was added later will overlap older labels.',
                        content=segmentation_mode_checkbox,
                    ),
                    Field(
                        title="IoU threshold",
                        description="Is used to match objects. IoU - Intersection over Union.",
                        content=threshold_input,
                    ),
                    compare_btn,
                ]
            ),
        ),
        Card(
            title="4️⃣ Compare Results",
            description="Click on 'CALCULATE CONSENSUS' button to see comparison matrix. Value in a table cell is a consensus score between two users",
            content=Container(
                widgets=[
                    report_progress_current_pair,
                    report_calculation_progress,
                    result_table,
                ]
            ),
        ),
        actions_card,
        consensus_report_notification,
        consensus_report_error_notification,
        consensus_report_text,
        consensus_report_details,
        report_layout,
    ]
)
