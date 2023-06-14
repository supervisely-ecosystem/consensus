import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
server_address = sly.env.server_address()
api_token = sly.env.api_token()
team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)

api = sly.Api()

workspace = api.workspace.get_info_by_id(workspace_id)
team = api.team.get_info_by_id(team_id)
all_users = {user.id: user for user in api.user.get_team_members(team_id)}
if project_id:
    all_projects = {project_id: api.project.get_info_by_id(project_id)}
else:
    all_projects = {
        project.id: project for project in api.project.get_list(workspace_id)
    }

if dataset_id:
    all_datasets = {dataset_id: api.dataset.get_info_by_id(dataset_id)}
else:
    all_datasets = {
        dataset.id: dataset
        for project in all_projects.values()
        for dataset in api.dataset.get_list(project.id)
    }

project_metas = {
    project.id: sly.ProjectMeta.from_json(api.project.get_meta(project.id))
    for project in all_projects.values()
}

all_img_infos = {}
ds_img_ids = {}
ann_infos = {}


def get_ds_ann_infos(dataset_id) -> list[sly.api.annotation_api.AnnotationInfo]:
    global ann_infos
    if dataset_id not in ann_infos:
        ann_infos[dataset_id] = api.annotation.get_list(dataset_id)
    return ann_infos[dataset_id]


def get_ds_img_infos(dataset_id) -> list[sly.ImageInfo]:
    global ds_img_ids
    global all_img_infos
    if dataset_id not in ds_img_ids:
        imgs = api.image.get_list(dataset_id)
        for img in imgs:
            all_img_infos[img.id] = img
        ds_img_ids[dataset_id] = [img.id for img in imgs]
    return [all_img_infos[img_id] for img_id in ds_img_ids[dataset_id]]
