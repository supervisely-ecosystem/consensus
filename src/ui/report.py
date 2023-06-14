import supervisely as sly
from supervisely.app.widgets import (
    Container,
    Text,
    Card,
    Table,
    GridGallery,
)
from supervisely.app import DataJson

gt_images = []
pred_images = []
gt_annotations = []
pred_annotations = []
differences = []
left_name = ""
right_name = ""

overall_score = Text("")
overall_stats = Card(title="Overall Score", content=overall_score)

obj_count_per_class_table_columns = [
    "NAME",
    "GT Objects",
    "Labeled Objects",
    "Recall(Matched objects)",
    "Precision",
    "F-measure",
]
obj_count_per_class_table = Table(columns=obj_count_per_class_table_columns)
obj_count_per_class_last = Text()
obj_count_per_class = Card(
    title="OBJECTS COUNT PER CLASS",
    content=Container(
        widgets=[obj_count_per_class_table, obj_count_per_class_last], gap=5
    ),
)

geometry_quality_table_columns = ["NAME", "Pixel Accuracy", "IOU"]
geometry_quality_table = Table(columns=geometry_quality_table_columns)
geometry_quality_last = Text()
geometry_quality = Card(
    title="GEOMETRY QUALITY",
    content=Container(widgets=[geometry_quality_table, geometry_quality_last], gap=5),
)

tags_stat_table_columns = [
    "NAME",
    "GT Tags",
    "Labeled Tags",
    "Precision",
    "Recall",
    "F-measure",
]
tags_stat_table = Table(columns=tags_stat_table_columns)
tags_stat_last = Text()
tags_stat = Card(
    title="TAGS", content=Container(widgets=[tags_stat_table, tags_stat_last], gap=5)
)

report_per_image_table_columns = [
    "NAME",
    "Objects Score",
    "Objects Missing",
    "Objects False Positive",
    "Tags Score",
    "Tags Missing",
    "Tags False Positive",
    "Geometry Score",
    "Overall Score",
]
report_per_image_table = Table(columns=report_per_image_table_columns)


@report_per_image_table.click
def show_images(datapoint):
    global report_per_image_images
    report_per_image_images.clean_up()
    global gt_images
    global pred_images
    global gt_annotations
    global pred_annotations
    global differences
    global left_name
    global right_name

    row = datapoint.row
    img_name = row["NAME"]

    img_idx = None
    for i, img in enumerate(gt_images):
        if img.name == img_name:
            img_idx = i
            break
    if img_idx is None:
        return

    gt_img = gt_images[img_idx]
    gt_img: sly.ImageInfo
    labels = (
        [
            sly.Label(
                differences[img_idx],
                sly.ObjClass("difference", sly.Bitmap, (255, 0, 0)),
            )
        ]
        if differences[img_idx] is not None
        else []
    )
    diff_ann = sly.Annotation(
        img_size=(gt_img.height, gt_img.width),
        labels=labels,
    )

    report_per_image_images.append(
        gt_img.preview_url, gt_annotations[img_idx], title=left_name, column_index=0
    )
    report_per_image_images.append(
        pred_images[img_idx].preview_url,
        pred_annotations[img_idx],
        title=right_name,
        column_index=1,
    )
    report_per_image_images.append(
        gt_img.preview_url, diff_ann, title="Difference", column_index=2
    )

    DataJson().send_changes()


report_per_image_images = GridGallery(3)
report_per_image = Card(
    title="REPORT PER IMAGE",
    description="Click on a row to see annotation differences",
    content=Container(
        widgets=[
            report_per_image_table,
            Card(content=report_per_image_images, collapsable=True),
        ]
    ),
)

results = Container(
    widgets=[
        overall_stats,
        obj_count_per_class,
        geometry_quality,
        tags_stat,
        report_per_image,
    ],
    gap=10,
)
layout = results


def get_overall_score(result):
    for data in result:
        if data["metric_name"] == "overall-score":
            if (
                data["class_gt"] == ""
                and data["image_gt_id"] == 0
                and data["tag_name"] == ""
            ):
                return data["value"]
    return 0


def get_obj_count_per_class_row(result, class_name):
    num_objects_gt = 0
    num_objects_pred = 0
    matches_recall_percent = 1
    matches_precision_percent = 1
    matches_f_measure = 1
    for data in result:
        if data["image_gt_id"] == 0:
            if (
                data["metric_name"] == "num-objects-gt"
                and data["class_gt"] == class_name
            ):
                num_objects_gt = data["value"]
            if (
                data["metric_name"] == "num-objects-pred"
                and data["class_gt"] == class_name
            ):
                num_objects_pred = data["value"]
            if (
                data["metric_name"] == "matches-recall"
                and data["class_gt"] == class_name
            ):
                matches_recall_percent = data["value"]
            if (
                data["metric_name"] == "matches-precision"
                and data["class_gt"] == class_name
            ):
                matches_precision_percent = data["value"]
            if data["metric_name"] == "matches-f1" and data["class_gt"] == class_name:
                matches_f_measure = data["value"]
    return [
        class_name,
        str(num_objects_gt),
        str(num_objects_pred),
        f"{int(matches_recall_percent*num_objects_gt)} of {num_objects_gt} ({round(matches_recall_percent*100, 2)}%)",
        f"{int(matches_precision_percent*num_objects_pred)} of {num_objects_pred} ({round(matches_precision_percent*100, 2)}%)",
        f"{round(matches_f_measure*100, 2)}%",
    ]


def get_average_f_measure_per_class(result):
    avg_f1 = 1
    f1_measures = []
    for data in result:
        if data["image_gt_id"] == 0:
            if data["metric_name"] == "matches-f1" and data["class_gt"] != "":
                f1_measures.append(data["value"])
    if len(f1_measures) > 0:
        avg_f1 = sum(f1_measures) / len(f1_measures)
    return avg_f1


def get_geometry_quality_row(result, class_name):
    pixel_accuracy = 1
    iou = 1
    for data in result:
        if data["image_gt_id"] == 0:
            if (
                data["metric_name"] == "pixel-accuracy"
                and data["class_gt"] == class_name
            ):
                pixel_accuracy = data["value"]
            if data["metric_name"] == "iou" and data["class_gt"] == class_name:
                iou = data["value"]

    return [class_name, f"{round(pixel_accuracy*100, 2)}%", f"{round(iou*100, 2)}%"]


def get_average_iou(result):
    avg_iou = 1
    iou = []
    for data in result:
        if data["image_gt_id"] == 0:
            if data["metric_name"] == "iou" and data["class_gt"] != "":
                iou.append(data["value"])
    if len(iou) > 0:
        avg_iou = sum(iou) / len(iou)
    return avg_iou


def get_tags_stat_table_row(result, tag_name):
    total_gt = 0
    total_pred = 0
    precision = 1
    recall = 1
    f1 = 1
    for data in result:
        if data["image_gt_id"] == 0:
            if data["metric_name"] == "tags-total-gt" and data["tag_name"] == tag_name:
                total_gt = data["value"]
            if (
                data["metric_name"] == "tags-total-pred"
                and data["tag_name"] == tag_name
            ):
                total_pred = data["value"]
            if data["metric_name"] == "tags-precision" and data["tag_name"] == tag_name:
                precision = data["value"]
            if data["metric_name"] == "tags-recall" and data["tag_name"] == tag_name:
                recall = data["value"]
            if data["metric_name"] == "tags-f1" and data["tag_name"] == tag_name:
                f1 = data["value"]

    return [
        tag_name,
        total_gt,
        total_pred,
        f"{int(precision*total_pred)} of {total_pred} ({round(precision*100, 2)}%)",
        f"{int(recall*total_gt)} of {total_gt} ({round(recall*100, 2)}%)",
        f"{round(f1*100, 2)}%",
    ]


def get_average_f_measure_per_tags(result):
    avg_f1 = 1
    f1_measures = []
    for data in result:
        if data["image_gt_id"] == 0:
            if data["metric_name"] == "tags-f1" and data["tag_name"] == "":
                f1_measures.append(data["value"])
    if len(f1_measures) > 0:
        avg_f1 = sum(f1_measures) / len(f1_measures)
    return avg_f1


def get_report_per_image_row(result, image_name, image_id):
    objects_score = 1
    objects_missing = None
    obj_false_positive = None
    tag_score = 1
    tag_missing = None
    tag_false_positive = None
    geometry_score = 0
    overall_score = 0
    for data in result:
        if (
            data["image_gt_id"] == image_id
            and data["class_gt"] == ""
            and data["tag_name"] == ""
        ):
            if data["metric_name"] == "matches-f1":
                objects_score = data["value"]
            if data["metric_name"] == "matches-false-negative":
                objects_missing = data["value"]
            if data["metric_name"] == "matches-false-positive":
                obj_false_positive = data["value"]
            if data["metric_name"] == "tags-f1":
                tag_score = data["value"]
            if data["metric_name"] == "tags-false-negative":
                tag_missing = data["value"]
            if data["metric_name"] == "tags-false-positive":
                tag_false_positive = data["value"]
            if data["metric_name"] == "iou":
                geometry_score = data["value"]
            if data["metric_name"] == "overall-score":
                overall_score = data["value"]

    return [
        image_name,
        f"{round(objects_score*100, 2)}%",
        objects_missing,
        obj_false_positive,
        f"{round(tag_score*100, 2)}%",
        tag_missing,
        tag_false_positive,
        f"{round(geometry_score*100, 2)}%",
        f"{round(overall_score*100, 2)}%",
    ]


def clean_up():
    report_per_image_images.clean_up()
    obj_count_per_class_table.read_json(
        {
            "columns": obj_count_per_class_table_columns,
            "data": [],
        }
    )
    obj_count_per_class_last.set(
        text=f"<b>Objects score (average F-measure) {0.00}%</b>", status="text"
    )

    geometry_quality_table.read_json(
        {
            "columns": geometry_quality_table_columns,
            "data": [],
        }
    )
    geometry_quality_last.set(
        f"<b>Geometry score (average IoU) {0.00}%</b>", status="text"
    )

    tags_stat_table.read_json(
        {
            "columns": tags_stat_table_columns,
            "data": [],
        }
    )
    tags_stat_last.set("<b>Tags score (average F-measure) 0%</b>", status="text")

    report_per_image_table.read_json(
        {
            "columns": report_per_image_table_columns,
            "data": [],
        }
    )

    overall_score.set("-", status="text")


def render_report(
    report,
    gt_imgs,
    pred_imgs,
    gt_anns,
    pred_anns,
    diffs,
    classes,
    tags,
    first_name,
    second_name,
):
    results.loading = True
    report_per_image_images.clean_up()

    global gt_images
    global pred_images
    global gt_annotations
    global pred_annotations
    global differences
    global left_name
    global right_name

    gt_images = gt_imgs
    pred_images = pred_imgs
    gt_annotations = gt_anns
    pred_annotations = pred_anns
    differences = diffs
    left_name = first_name
    right_name = second_name

    # overall score
    def get_score_text(score):
        if score > 0.66:
            return f'<span style="color: green"><h2>{round(score*100, 2)}</h2></span>'
        if score > 0.33:
            return f'<span style="color: orange"><h2>{round(score*100, 2)}</h2></span>'
        return f'<span style="color: red"><h2>{round(score*100, 2)}</h2></span>'

    overall_score.set(get_score_text(get_overall_score(report)), status="text")

    # obj count per class
    obj_count_per_class_table.read_json(
        {
            "columns": obj_count_per_class_table_columns,
            "data": [
                get_obj_count_per_class_row(report, cls_name) for cls_name in classes
            ],
        }
    )
    obj_count_per_class_last.set(
        text=f"<b>Objects score (average F-measure) {round(get_average_f_measure_per_class(report)*100, 2)}%</b>",
        status="text",
    )

    # geometry quality
    geometry_quality_table.read_json(
        {
            "columns": geometry_quality_table_columns,
            "data": [
                get_geometry_quality_row(report, cls_name) for cls_name in classes
            ],
        }
    )
    geometry_quality_last.set(
        f"<b>Geometry score (average IoU) {round(get_average_iou(report)*100, 2)}%</b>",
        status="text",
    )

    # tags
    tags_stat_table.read_json(
        {
            "columns": tags_stat_table_columns,
            "data": [get_tags_stat_table_row(report, tag_name) for tag_name in tags],
        }
    )
    tags_stat_last.set(
        f"<b>Tags score (average F-measure) {round(get_average_f_measure_per_tags(report)*100, 2)}%</b>",
        status="text",
    )

    # per image
    report_per_image_table.read_json(
        {
            "columns": report_per_image_table_columns,
            "data": [
                get_report_per_image_row(report, gt_img.name, gt_img.id)
                for gt_img in gt_imgs
            ],
        }
    )

    results.loading = False
