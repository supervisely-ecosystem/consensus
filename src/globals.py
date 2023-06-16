import os
from typing import List
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
        project.id: project for project in api.project.get_list(workspace_id) if project.type == str(sly.ProjectType.IMAGES)
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


def get_ds_ann_infos(dataset_id) -> List[sly.api.annotation_api.AnnotationInfo]:
    return api.annotation.get_list(dataset_id)


def get_ds_img_infos(dataset_id) -> List[sly.ImageInfo]:
    return api.image.get_list(dataset_id)
