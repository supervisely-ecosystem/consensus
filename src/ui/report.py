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


def report_to_dict(report):
    d = {}
    for metric in report:
        if metric["metric_name"] not in d:
            d[metric["metric_name"]] = {}
        if metric["image_gt_id"] not in d[metric["metric_name"]]:
            d[metric["metric_name"]][metric["image_gt_id"]] = {
                "image_pred_id": metric["image_pred_id"]
            }
        d[metric["metric_name"]][metric["image_gt_id"]][(metric["class_gt"], metric["tag_name"])] = metric["value"]
    return d


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
    try:
        labels = [sly.Label(differences[img_idx], sly.ObjClass("difference", sly.Bitmap, (255, 0, 0)))]
    except:
        labels = []
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


def get_overall_score(report):
    try:
        return report["overall-score"][0][("","")]
    except KeyError:
        return 0


def get_obj_count_per_class_row(report, class_name):
    metrics = {
        "num-objects-gt": 0,
        "num-objects-pred": 0,
        "matches-recall": 1,
        "matches-precision": 1,
        "matches-f1": 1,
    }
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = report[metric_name][0][(class_name, "")]
        except KeyError:
            pass
    return [
        class_name,
        str(metrics["num-objects-gt"]),
        str(metrics["num-objects-pred"]),
        f'{int(metrics["matches-recall"]*metrics["num-objects-gt"])} of {metrics["num-objects-gt"]} ({round(metrics["matches-recall"]*100, 2)}%)',
        f'{int(metrics["matches-precision"]*metrics["num-objects-pred"])} of {metrics["num-objects-pred"]} ({round(metrics["matches-precision"]*100, 2)}%)',
        f'{round(metrics["matches-f1"]*100, 2)}%',
    ]


def get_average_f_measure_per_class(report):
    try:
        return report["matches-f1"][0][("","")]
    except KeyError:
        return 1


def get_geometry_quality_row(report, class_name):
    metrics = {
        "pixel-accuracy": 1,
        "iou": 1,
    }
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = report[metric_name][0][(class_name, "")]
        except KeyError:
            pass

    return [class_name, f'{round(metrics["pixel-accuracy"]*100, 2)}%', f'{round(metrics["iou"]*100, 2)}%']


def get_average_iou(report):
    try:
        return report["iou"][0][("","")]
    except KeyError:
        return 1


def get_tags_stat_table_row(report, tag_name):
    metrics = {
        "tags-total-gt": 0,
        "tags-total-pred": 0,
        "tags-precision": 1,
        "tags-recall": 1,
        "tags-f1": 1,
    }
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = report[metric_name][0][("", tag_name)]
        except KeyError:
            pass
    return [
        tag_name,
        metrics["tags-total-gt"],
        metrics["tags-total-pred"],
        f'{int(metrics["tags-precision"]*metrics["tags-total-pred"])} of {metrics["tags-total-pred"]} ({round(metrics["tags-precision"]*100, 2)}%)',
        f'{int(metrics["tags-recall"]*metrics["tags-total-gt"])} of {metrics["tags-total-gt"]} ({round(metrics["tags-recall"]*100, 2)}%)',
        f'{round(metrics["tags-f1"]*100, 2)}%',
    ]


def get_average_f_measure_per_tags(report):
    try:
        return report["tags-f1"][0][("", "")]
    except KeyError:
        return 1


def get_report_per_image_row(report, image_name, image_id):
    metrics = {
        "matches-f1": 0.0,
        "matches-false-negative": 0,
        "matches-false-positive": 0,
        "tags-f1": 0.0,
        "tags-false-negative": 0,
        "tags-false-positive": 0,
        "iou": 0.0,
        "overall-score": 0.0,
    }
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = report[metric_name][image_id][("", "")]
        except KeyError:
            pass
    return [
        image_name,
        f'{round(metrics["matches-f1"]*100, 2)}%',
        metrics["matches-false-negative"],
        metrics["matches-false-positive"],
        f'{round(metrics["tags-f1"]*100, 2)}%',
        metrics["tags-false-negative"],
        metrics["tags-false-positive"],
        f'{round(metrics["iou"]*100, 2)}%',
        f'{round(metrics["overall-score"]*100, 2)}%',
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

    report = report_to_dict(report)

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
