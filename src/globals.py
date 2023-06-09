import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
server_address = sly.env.server_address()
api_token = sly.env.api_token()
team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()

api = sly.Api()

all_projects = {project.id: project for project in api.project.get_list(workspace_id)}
project_metas = {project.id: sly.ProjectMeta.from_json(api.project.get_meta(project.id)) for project in all_projects.values()}
all_datasets = {dataset.id: dataset for project in all_projects.values() for dataset in api.dataset.get_list(project.id)}
ann_infos = {}

def get_ann_infos(dataset_id) -> list[sly.api.annotation_api.AnnotationInfo]:
    global ann_infos
    if dataset_id not in ann_infos:
        ann_infos[dataset_id] = api.annotation.get_list(dataset_id)
    return ann_infos[dataset_id]
