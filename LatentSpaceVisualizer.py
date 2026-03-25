"""
A web browser-based application to visualize and interactively explore the latent spaces
created with the (variational) autencoders in this repository.

Author: Pepe Eulzer
Year: 2026
Publication: [UNDER REVIEW - add title / DOI]
License: see repository license

Dependencies:
* numpy
* pandas
* pyvista
* pytorch
* trame
* vtk

Usage:
* configure DATA_PARENT_DIR (requires to contain faces_<vertex-count>.txt)
* configure LABELS_PATH to point to a csv file with all desired metadata (including a 'status' column)
* for comparing to ground truth meshes, provide obj files in 'DATA_PARENT_DIR/<vertex-count>_vertices'
* run the script

Build with trame: https://github.com/Kitware/trame
"""

import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import plotly.graph_objects as go

import pyvista as pv

from trame.app import get_server
from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.widgets import vuetify, plotly, trame, vtk as vtk_widgets

from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)
from vtkmodules.vtkRenderingFreeType import vtkVectorText
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa
import vtkmodules.vtkRenderingOpenGL2  # noqa

from src.dataset import AneurysmDataset
from src.pointnet_ae import AEDecoder, AEEncoder
from src.pointnet_vae import VAEDecoder, VAEEncoder

# Defaults
pv.OFF_SCREEN = True
CURRENT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
class ModelType:
    ae = 0   # Plain Autoencoder (AE)
    vae = 1  # Variational Autoencoder (VAE)

MODEL_TYPE_STRINGS = {
    ModelType.ae: "AE",
    ModelType.vae: "VAE",
}

class ModelResolution:
    low = 0
    high = 1
    very_high = 2
    values_int = [716, 2956, 12156]
    values_str = ["716", "2956", "12156"]

DATA_PARENT_DIR = os.path.join(CURRENT_DIRECTORY, 'data')
LABELS_PATH = os.path.join(DATA_PARENT_DIR, 'labels_merged_cleaned.csv')

LOCAL_LINSPACE_RES = 91

# -----------------------------------------------------------------------------
# Trame + state setup
# -----------------------------------------------------------------------------
server = get_server()
server.client_type = "vue2"
state, ctrl = server.state, server.controller
state.trame__title = "Aneurysm Latent Space"
state.model_type = ModelType.vae      # controls the displayed model type

# AE/VAE state
state.show_ae_vae_ui = True           # True: shows UI elements to control AE/VAE models
state.out_dim = ModelResolution.high  # controls the mesh resolution in AE and VAE models

# interaction/view settings
state.hover_select = False            # True: selected 2D point is updated on hover event, False: click event
state.snap_to_points = False          # True: selected 2D point snaps to input model points
state.show_ground_truth = False       # True: the ground truth mesh is shown (if snapping is enabled)
state.cube_axes_visibility = False    # True: shows cube axes around 3D mesh

# encoder/decoder settings (Python only -> not in state)
z_size = 2
use_bias = True
device = "cpu" # one of ["cpu", "cuda"]

# initialized later
model_decoder = None
data_directory = ""
import_dir = ""

# -----------------------------------------------------------------------------
# Pytorch setup
# -----------------------------------------------------------------------------

def get_latent_space_points(encoder_path, data_dir, labels_file_path):
    # intialize encoder
    if state.model_type in [ModelType.ae]:
        model_encoder = AEEncoder(z_size=z_size, use_bias=use_bias).to(device)
    elif state.model_type in [ModelType.vae]:
        model_encoder = VAEEncoder(z_size=z_size, use_bias=use_bias).to(device)
    else:
        raise Exception("Encoder cannot be initialized with a non-autoencoder type model.")
    
    # load checkpoint, disable gradients
    model_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    model_encoder.eval()

    # load dataset
    aneurysm_dataset = AneurysmDataset(
        data_dir,
        labels_file_path,
        split="train",
        train_split_percentage=1.0, # shows all data
    )
    
    # setup dataloader
    aneurysm_data_loader = DataLoader(
        aneurysm_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True
    )

    point_coordinates = []
    for point_data, label_data in aneurysm_data_loader:
        X = point_data.to(device)
        
        # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
        if X.size(-1) == 3:
            X.transpose_(X.dim() - 2, X.dim() - 1)

        # run encoder
        if state.model_type in [ModelType.ae]:
            z = model_encoder(X)
        elif state.model_type in [ModelType.vae]:
            z, _, _ = model_encoder(X)

        # dissassemble batch
        point_coordinates.append(z.tolist()[0])

    return np.array(point_coordinates)

def get_decoder(decoder_path, z_size, use_bias, out_dim, device):
    if state.model_type in [ModelType.ae]:
        model_decoder = AEDecoder(z_size=z_size, use_bias=use_bias, out_dim=out_dim).to(device)
    elif state.model_type in [ModelType.vae]:
        model_decoder = VAEDecoder(z_size=z_size, use_bias=use_bias, out_dim=out_dim).to(device)
    else:
        raise Exception("Decoder cannot be initialized with a non-autoencoder type model.")
    
    # load checkpoint, disable gradients
    model_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    model_decoder.eval()

    return model_decoder

def generate_from_coordinate(target_point_coordinate, model_decoder):
    target_point_tensor = torch.tensor(target_point_coordinate, dtype=torch.float32).unsqueeze(0).to(device)

    # Perform the forward pass
    with torch.no_grad():
        if state.model_type in [ModelType.ae]:
            X_rec = model_decoder(target_point_tensor)
        elif state.model_type in [ModelType.vae]:
            X_rec = model_decoder((target_point_tensor, None, None))
        else:
            raise Exception("Decoder cannot be run with a non-autoencoder type model.")

    # Squeeze batch dimension out, swap the dimensions to have shape [nr vertices, 3]
    return X_rec.squeeze(0).transpose(0, 1).cpu().numpy()

# -----------------------------------------------------------------------------
# VTK setup
# -----------------------------------------------------------------------------

class SizeAnnotationBox():
    def __init__(self, renderer, visible=False):
        # outline (to visualize aneurysm size)
        self.outline = vtkOutlineFilter()
        self.outline.SetInputData(pv.PolyData())
        self.outline_mapper = vtkPolyDataMapper()
        self.outline_mapper.SetInputConnection(self.outline.GetOutputPort())
        self.outline_actor = vtkActor()
        self.outline_actor.SetMapper(self.outline_mapper)
        self.outline_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        self.outline_actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
        self.outline_actor.SetVisibility(visible)

        # outline texts
        self.texts = [vtkVectorText(), vtkVectorText(), vtkVectorText()] # x, y, z axis
        self.text_actors = [vtkActor(), vtkActor(), vtkActor()]
        self.transforms = [vtkTransform(), vtkTransform(), vtkTransform()]
        self.transforms[0].RotateX(45)
        self.transforms[1].RotateY(-45)
        self.transforms[2].RotateZ(-45)
        self.transforms[1].RotateZ(270)
        self.transforms[2].RotateY(90)
        for i in range(3):
            self.texts[i].SetText("\nTest")
            transform_filter = vtkTransformFilter()
            transform_filter.SetInputConnection(self.texts[i].GetOutputPort())
            transform_filter.SetTransform(self.transforms[i])
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(transform_filter.GetOutputPort())
            self.text_actors[i].SetMapper(mapper)
            self.text_actors[i].SetScale(0.1, 0.1, 0.1)
            self.text_actors[i].SetPosition(0.0, 0.0, 0.0)
            self.text_actors[i].GetProperty().SetColor(0.0, 0.0, 0.0)
            self.text_actors[i].SetVisibility(visible)
        
        renderer.AddActor(self.outline_actor)
        for actor in self.text_actors:
            renderer.AddActor(actor)

    def setVisible(self, visible):
        self.outline_actor.SetVisibility(visible)
        for actor in self.text_actors:
            actor.SetVisibility(visible)
    
    def updateLabels(self, b):
        x_width = b[1] - b[0]
        y_width = b[3] - b[2]
        z_width = b[5] - b[4]
        self.texts[0].SetText("\n" + "{:.1f}".format(x_width) + " mm")
        self.texts[1].SetText("\n" + "{:.1f}".format(y_width) + " mm")
        self.texts[2].SetText("\n" + "{:.1f}".format(z_width) + " mm")
        s = np.sqrt(x_width**2 + y_width**2 + z_width**2) * 0.01
        self.text_actors[0].SetScale(s, s, s)
        self.text_actors[1].SetScale(s, s, s)
        self.text_actors[2].SetScale(s, s, s)
        box_bounds = self.text_actors[0].GetBounds()
        w = 0.5 * (box_bounds[1] - box_bounds[0])
        self.text_actors[0].SetPosition(b[0]+0.5*x_width-w, b[2], b[4])
        self.text_actors[1].SetPosition(b[0], b[2]+0.5*y_width+w, b[4])
        self.text_actors[2].SetPosition(b[0], b[2], b[4]+0.5*z_width+w)

renderer = vtkRenderer()
renderer.SetBackground(1.0,1.0,1.0)
render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)

render_window_interactor = vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)
render_window_interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

mesh_mapper = vtkPolyDataMapper()
mesh_mapper.SetInputData(pv.PolyData())
mesh_actor = vtkActor()
mesh_actor.SetMapper(mesh_mapper)
mesh_actor.GetProperty().EdgeVisibilityOn()

# red phong shading
mesh_actor.GetProperty().SetColor(195/255, 22/255, 27/255)
mesh_actor.GetProperty().SetInterpolationToPhong()
mesh_actor.GetProperty().SetAmbient(0.3)
mesh_actor.GetProperty().SetDiffuse(0.7)
mesh_actor.GetProperty().SetSpecular(0.5)
mesh_actor.GetProperty().SetSpecularPower(50)

ground_truth_mesh = pv.PolyData()
ground_truth_mesh_mapper = vtkPolyDataMapper()
ground_truth_mesh_mapper.SetInputData(ground_truth_mesh)
ground_truth_mesh_actor = vtkActor()
ground_truth_mesh_actor.SetMapper(ground_truth_mesh_mapper)
ground_truth_mesh_actor.GetProperty().SetRepresentationToSurface()
ground_truth_mesh_actor.GetProperty().EdgeVisibilityOn()
ground_truth_mesh_actor.GetProperty().SetColor(1, 1, 1)
ground_truth_mesh_actor.GetProperty().SetOpacity(0.6)
ground_truth_mesh_actor.GetProperty().SetEdgeColor(0, 0, 0)
ground_truth_mesh_actor.SetVisibility(state.show_ground_truth)

size_annotation_box = SizeAnnotationBox(renderer, state.cube_axes_visibility)

renderer.AddActor(mesh_actor)
renderer.AddActor(ground_truth_mesh_actor)
renderer.ResetCamera()

# -----------------------------------------------------------------------------
# Plotly setup
# -----------------------------------------------------------------------------
fig = go.Figure()
dragmode = None
latent_space_df = pd.DataFrame()
import_df = pd.DataFrame()
local_x_linspace = np.linspace(-1, 1, LOCAL_LINSPACE_RES)
local_y_linspace = np.linspace(-1, 1, LOCAL_LINSPACE_RES)
local_z_heatmap = np.zeros(shape=(local_y_linspace.size, local_x_linspace.size), dtype=np.byte)
global_x_linspace = np.linspace(-10, 10, 10)
global_y_linspace = np.linspace(-10, 10, 10)
global_z_heatmap = np.zeros(shape=(global_y_linspace.size, global_x_linspace.size), dtype=np.byte)
ls_x = 0
ls_y = 0

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@state.change("cube_axes_visibility")
def update_cube_axes_visibility(cube_axes_visibility, **kwargs):
    size_annotation_box.setVisible(cube_axes_visibility)
    size_annotation_box.updateLabels(mesh_actor.GetBounds())
    ctrl.update_view()

def relayout(event):
    global local_x_linspace, local_y_linspace, dragmode
    if 'dragmode' in event:
        dragmode = event['dragmode']
    elif dragmode == 'zoom' and 'xaxis.range[0]' in event:
        rx = 0.05 * (event['xaxis.range[1]'] - event['xaxis.range[0]'])
        ry = 0.05 * (event['yaxis.range[1]'] - event['yaxis.range[0]'])
        local_x_linspace = np.linspace(-rx, rx, LOCAL_LINSPACE_RES)
        local_y_linspace = np.linspace(-ry, ry, LOCAL_LINSPACE_RES)
        if not state.snap_to_points:
            fig['data'][-2]['x'] = local_x_linspace + ls_x
            fig['data'][-2]['y'] = local_y_linspace + ls_y
            fig['data'][0]['x'] = np.linspace(event['xaxis.range[0]'], event['xaxis.range[1]'], 20)
            fig['data'][0]['y'] = np.linspace(event['yaxis.range[0]'], event['yaxis.range[1]'], 20)
            ctrl.update_figure(fig)
    elif 'xaxis.autorange' in event:
        local_x_linspace = np.linspace(-1, 1, LOCAL_LINSPACE_RES)
        local_y_linspace = np.linspace(-1, 1, LOCAL_LINSPACE_RES)
        if not state.snap_to_points:
            fig['data'][-2]['x'] = local_x_linspace + ls_x
            fig['data'][-2]['y'] = local_y_linspace + ls_y
            fig['data'][0]['x'] = global_x_linspace
            fig['data'][0]['y'] = global_y_linspace
            ctrl.update_figure(fig)

def set_hover_select(hover_select):
    state.hover_select = hover_select

@state.change("snap_to_points")
def set_snap_to_points(snap_to_points, **kwargs):
    if not snap_to_points and state.show_ground_truth:
        state.show_ground_truth = False

@state.change("show_ground_truth")
def set_ground_truth(show_ground_truth, **kwargs):
    global ground_truth_mesh
    visible = ground_truth_mesh_actor.GetVisibility()
    if show_ground_truth and not visible:
        if not os.path.isdir(data_directory):
            print("Data dir cannot be found:", data_directory)
            ground_truth_mesh_actor.SetVisibility(0)
            return
        selection = latent_space_df[latent_space_df['x'] == ls_x]['dataset']
        if len(selection == 1):
            filename = selection.iloc[0]
            ground_truth_mesh = pv.PolyData(os.path.join(data_directory, filename))
            ground_truth_mesh_mapper.SetInputData(ground_truth_mesh)
            ground_truth_mesh_actor.SetVisibility(1)
            ctrl.update_view()
    elif not show_ground_truth and visible:
        ground_truth_mesh_actor.SetVisibility(0)
        ctrl.update_view()

def update_mesh(x, y):
    global ground_truth_mesh, ls_x, ls_y
    ls_x = x[0]
    ls_y = y[0]
    mesh.points = generate_from_coordinate([ls_x, ls_y], model_decoder)

    if state.show_ground_truth:
        selection = latent_space_df[np.logical_and(latent_space_df['x'] == ls_x, latent_space_df['y'] == ls_y)]['dataset']
        if import_df.size > 0:
            selection_import = import_df[np.logical_and(import_df['x'] == ls_x, import_df['y'] == ls_y)]['dataset']
            if len(selection_import == 1):
                filename = selection_import.iloc[0]
                ground_truth_mesh = pv.PolyData(os.path.join(import_dir, filename))
                ground_truth_mesh_mapper.SetInputData(ground_truth_mesh)
                ground_truth_mesh_actor.SetVisibility(1)
        
        if len(selection == 1):
            filename = selection.iloc[0]
            ground_truth_mesh = pv.PolyData(os.path.join(data_directory, filename))
            ground_truth_mesh_mapper.SetInputData(ground_truth_mesh)
            ground_truth_mesh_actor.SetVisibility(1)
        elif not(import_df.size > 0 and len(selection_import == 1)):
            # no ground truth or ambiguous selection
            ground_truth_mesh_actor.SetVisibility(0)

    if state.cube_axes_visibility:
            size_annotation_box.updateLabels(mesh_actor.GetBounds())
    ctrl.update_view()

    if not state.snap_to_points:
        fig['data'][-2]['x'] = local_x_linspace + ls_x
        fig['data'][-2]['y'] = local_y_linspace + ls_y
    fig['data'][-1]['x'] = [ls_x]
    fig['data'][-1]['y'] = [ls_y]
    ctrl.update_figure(fig)

def update_mesh_hover(x, y):
    if state.hover_select:
        update_mesh(x, y)
    
def update_mesh_click(x, y):
    if state.hover_select:
        state.hover_select = False
    else:
        update_mesh(x, y)

def __get_labels(df):
    # helper method for load_figure
    labels = df['dataset'].to_list()
    for i in range(len(df.index)):
        hospital = str(df['hospital'].iloc[i])
        if hospital != 'nan':
            labels[i] += "<br>hospital: " + hospital

        location = str(df['location'].iloc[i])
        if location != 'nan':
            labels[i] += "<br>location: " + location
        
        sex = str(df['sex'].iloc[i])
        if sex != 'nan':
            labels[i] += "<br>sex: " + sex

        age = str(df['age'].iloc[i])
        if age != 'nan':
            labels[i] += "<br>age: " + age

        status = str(df['status'].iloc[i])
        if status != 'nan':
            labels[i] += "<br>status: " + status
        else:
            labels[i] += "<br>status: unknown"

    return labels
    
@state.change("figure_size", "snap_to_points")
def load_figure(figure_size, snap_to_points, **kwargs):
    # figure color scheme
    # https://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5

    if figure_size is None:
        return

    ruptured = latent_space_df[latent_space_df['status'] == 'ruptured']
    unknown = latent_space_df[latent_space_df['status'] != 'ruptured']

    # unruptured_labels = __get_labels(unruptured)
    ruptured_labels = __get_labels(ruptured)
    unknown_labels = __get_labels(unknown)
    if import_df.size > 0:
        import_labels = __get_labels(import_df)

    global fig
    fig = go.Figure()

    # global hoverable background heatmap
    if not snap_to_points:
        fig.add_trace(
            go.Heatmap(
                x=global_x_linspace,
                y=global_y_linspace,
                z=global_z_heatmap,
                colorscale=[[0.0, "rgba(255, 255, 255, 0)"], [1.0, "rgba(255, 255, 255, 0)"]],
                showscale=False,
                hoverinfo='none',
                hoverongaps = False
            )
        )

    fig.add_trace(
        go.Scatter(
            name="other",
            x=unknown['x'],
            y=unknown['y'],
            text=unknown_labels,
            mode='markers',
            marker=dict(color='#4C72B0')
        )
    )

    fig.add_trace(  
        go.Scatter(
            name="ruptured",
            x=ruptured['x'],
            y=ruptured['y'],
            text=ruptured_labels,
            mode='markers',
            marker=dict(color='#DD8452')
        )
    )

    if import_df.size > 0:
        fig.add_trace(  
            go.Scatter(
                name="imported",
                x=import_df['x'],
                y=import_df['y'],
                text=import_labels,
                mode='markers',
                marker=dict(size=10, color='#ffffbf', symbol="diamond", line=dict(width=1, color="DarkSlateGrey"))
            )
        )

    # local hoverable heatmap around cursor
    if not snap_to_points:
        fig.add_trace(
            go.Heatmap(
                x=local_x_linspace,
                y=local_y_linspace,
                z=local_z_heatmap,
                colorscale=[[0.0, "rgba(255, 255, 255, 0)"], [1.0, "rgba(255, 255, 255, 0)"]],
                showscale=False,
                hoverinfo='none',
                hoverongaps = False
            )
        )

    # circle marker, shows selected position
    fig.add_trace(
        go.Scatter(
            x=[ls_x], y=[ls_y],
            mode='markers',
            marker_symbol="circle-open",
            marker_color='#000000',
            marker_size=10,
            marker_line_width=3,
            hoverinfo='skip',
            showlegend=False
        )
    )

    # set the figure layout
    bounds = figure_size.get("size", {})
    fig.update_layout(
        autosize=False,
        width=bounds.get("width", 200),
        height=bounds.get("height", 200),
        showlegend=True,
        dragmode=False,
        uirevision=True
    )

    # Update chart
    ctrl.update_figure(fig)

def set_data_dir(size:str):
    # set ground truth obj files directory
    global data_directory
    data_directory = os.path.join(DATA_PARENT_DIR, size + '_vertices')
    if not os.path.exists(data_directory):
        print("WARNING: Data directory does not exist:", data_directory)

def get_encoder_decoder_paths(size:str):
    if state.model_type == ModelType.ae:
        encoder_path = os.path.join(CURRENT_DIRECTORY, "weights/pointnet_ae_" + size + "_pointwise_distance_model_encoder.pth")
        decoder_path = os.path.join(CURRENT_DIRECTORY, "weights/pointnet_ae_" + size + "_pointwise_distance_model_decoder.pth")
        cache_path   = os.path.join(CURRENT_DIRECTORY, "weights/pointnet_ae_" + size + "_pointwise_distance_cache.csv")
    elif state.model_type == ModelType.vae:
        encoder_path = os.path.join(CURRENT_DIRECTORY, "weights/pointnet_vae_" + size + "_pointwise_distance_kld_model_encoder.pth")
        decoder_path = os.path.join(CURRENT_DIRECTORY, "weights/pointnet_vae_" + size + "_pointwise_distance_kld_model_decoder.pth")
        cache_path   = os.path.join(CURRENT_DIRECTORY, "weights/pointnet_vae_" + size + "_pointwise_distance_kld_cache.csv")
    else:
        print("WARNING: Encoder/decoder paths requested but ModelType does not match.")
        encoder_path = ""
        encoder_path = ""
        cache_path = ""
    return encoder_path, decoder_path, cache_path

@state.change("model_type", "out_dim")
def set_model(model_type=state.model_type, out_dim=state.out_dim, **kwargs):
    global latent_space_df # will be configured, then all views will be reset
    global ls_x, ls_y # reset sample position to [0, 0]
    ls_x = 0
    ls_y = 0

    # ground truth model can only be shown *after* a point is selected
    ground_truth_mesh_actor.SetVisibility(0)

    # reset imports
    global import_df
    import_df = pd.DataFrame()

    # show corresponding UI
    state.show_ae_vae_ui = True

    # initilize encoder/decoder and mesh
    size = ModelResolution.values_str[out_dim]
    set_data_dir(size)
    mesh_actor.SetVisibility(1)
    ground_truth_mesh_actor.GetProperty().SetOpacity(0.6)
    encoder_path, decoder_path, cache_path = get_encoder_decoder_paths(size)
    
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print("WARNING: Encoder/Decoder path does not exist:", encoder_path)
        return

    # try to load latent space data frame from cache
    if os.path.exists(cache_path):
        latent_space_df = pd.read_csv(cache_path)
    else:
        # run all samples through encoder (loads all models, time intensive)
        try:
            print("No point cache found, running encoder...")
            latent_points_2D = get_latent_space_points(encoder_path, data_directory, LABELS_PATH)
        except Exception as e:
            print("Something went wrong. Could not load", data_directory, "due to the following exception:")
            print(e)
            return

        # cache result
        file_paths = [f for f in os.listdir(data_directory) if f.endswith(".obj")]
        latent_space_df = pd.DataFrame({
            'dataset':file_paths,
            'x':latent_points_2D[:,0],
            'y':latent_points_2D[:,1]
        })
        latent_space_df = pd.merge(latent_space_df, pd.read_csv(LABELS_PATH), how='left', on='dataset')
        print("Writing latent space data frame cache to", cache_path)
        latent_space_df.to_csv(cache_path, index=False)

    # initialize point cloud decoder and mesh
    global mesh, model_decoder
    model_decoder = get_decoder(decoder_path, z_size, use_bias, ModelResolution.values_int[out_dim], device)
    points = generate_from_coordinate([ls_x, ls_y], model_decoder)
    faces = np.loadtxt(os.path.join(DATA_PARENT_DIR, "faces_" + ModelResolution.values_str[out_dim] + ".txt"), dtype=np.int32)
    nr_faces = faces.shape[0]
    faces_flat = np.hstack((np.full((nr_faces, 1), 3), faces)).flat
    mesh = pv.PolyData(points, faces_flat, n_faces=nr_faces)
    mesh.compute_normals(inplace=True)
    mesh_mapper.SetInputData(mesh)
    size_annotation_box.outline.SetInputData(mesh)
    
    # build global heatmap (for plotly navigation)
    global global_x_linspace, global_y_linspace, global_z_heatmap
    global_x_linspace = np.linspace(latent_space_df['x'].min(), latent_space_df['x'].max(), 20)
    global_y_linspace = np.linspace(latent_space_df['y'].min(), latent_space_df['y'].max(), 20)
    global_z_heatmap = np.zeros(shape=(global_y_linspace.size, global_x_linspace.size), dtype=np.byte)

    if state.cube_axes_visibility:
        size_annotation_box.updateLabels(mesh_actor.GetBounds())

    # ctrl.reset_view_camera()
    renderer.ResetCamera()
    ctrl.update_view()

    # update chart with new data
    load_figure(state.figure_size, state.snap_to_points)

# -----------------------------------------------------------------------------
# GUI elements
# -----------------------------------------------------------------------------

CHART_STYLE = {
    "style": "position: absolute; left: 50%; transform: translateX(-50%);",
    "display_mode_bar": ("true",),
    "mode_bar_buttons_to_remove": (
        [
            "toImage",
            "resetScale2d",
            "zoomIn2d",
            "zoomOut2d",
            "select2d",
            "lasso2d",
            "toggleSpikelines",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
        ],
    ),
    "display_logo": ("false",),
    "scroll_zoom": ("false",),
}

def toolbar_buttons():
    vuetify.VCheckbox(
        v_model=("cube_axes_visibility"),
        on_icon="mdi-cube-outline",
        off_icon="mdi-cube-off-outline",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    with vuetify.VBtn(icon=True, click=ctrl.reset_view_camera):
        vuetify.VIcon("mdi-crop-free")

def ui_card(title):
    with vuetify.VCard():
        vuetify.VCardTitle(
            title,
            classes="grey lighten-1 py-1 grey--text text--darken-3",
            style="user-select: none;",
        )
        content = vuetify.VCardText(classes="py-2")
    return content

def enum_slider(v_model, v_show, tick_labels, label):
    return vuetify.VSlider(
        v_model=(v_model),
        v_show=(v_show),
        min=0,
        max=len(tick_labels)-1,
        step=1,
        tick_labels=(tick_labels,),
        label=label,
        ticks="always",
        tick_size="4",
    )

def model_card():
        with ui_card(title="Model"):
            vuetify.VSelect(
                v_model=("model_type"),
                items=(
                    [
                        {"text": "Autoencoder", "value": ModelType.ae},
                        {"text": "Variational Autoencoder", "value": ModelType.vae},
                    ],
                ),
                label="Model Type",
                outlined=True,
                classes="pt-1",
            )
            enum_slider("out_dim", "show_ae_vae_ui", ["700", "3k", "12k"], "Mesh Size")

def tools_card():
    with ui_card(title="Selection"):
        vuetify.VBtn(
            "Hover Select",
            v_model=("hover_select"),
            block="true",
            click=(set_hover_select, "[true]"),
            v_icon="mdi-plus"
        )
        vuetify.VCheckbox(
            v_model=("snap_to_points"),
            label="Snap to Points",
            on_icon="mdi-plus-lock",
            off_icon="mdi-plus-lock-open"
        )
        vuetify.VCheckbox(
            v_model=("show_ground_truth"),
            v_show=("snap_to_points"),
            label="Show Ground Truth",
            on_icon="mdi-database-eye",
            off_icon="mdi-database-eye-off"
        )

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("Aneurysm Latent Space")

    with layout.toolbar as tb:
        tb.dense = True
        vuetify.VSpacer()
        vuetify.VDivider(vertical=True, classes="mx-2")
        toolbar_buttons()

    with layout.drawer as drawer:
        drawer.width=325
        model_card()
        tools_card()

    with layout.content:
        with vuetify.VContainer(fluid=True, classes="fill-height pa-0 ma-0"):
            with vuetify.VRow(dense=True, style="height: 100%;"):
                with vuetify.VCol(
                    classes="pa-0",
                    style="border-right: 1px solid #ccc; position: relative;",
                ):
                    with trame.SizeObserver("figure_size"):
                        html_plot = plotly.Figure(
                            state_variable_name="html_plot",
                            hover=(
                                update_mesh_hover,
                                "[$event.points.map(({x}) => x), $event.points.map(({y}) => y)]"
                            ),
                            click=(
                                update_mesh_click,
                                "[$event.points.map(({x}) => x), $event.points.map(({y}) => y)]",
                            ),
                            relayout=(relayout, "[utils.safe($event)]"),
                            **CHART_STYLE)
                        ctrl.update_figure = html_plot.update
                with vuetify.VCol(classes="pa-0"):
                    view = vtk_widgets.VtkLocalView(
                        render_window,      # vtkRenderWindow instance
                        ref="vtk_view",     # identifier of this component
                        namespace="view",
                    )
                    ctrl.update_view = view.update
                    ctrl.reset_view_camera = view.reset_camera

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

ctrl.on_server_ready.add(set_model)

if __name__ == '__main__':
    # see https://trame.readthedocs.io/en/latest/core.server.html#trame_server.core.Server.start
    server.start(
        port=8080,
        open_browser=True,
        show_connection_info=True,
        timeout=0, # seconds to wait if no client is connected, 0 will disable auto-shutdown
        host='localhost' # 'localhost' will run locally, '0.0.0.0' will run on the web
    )