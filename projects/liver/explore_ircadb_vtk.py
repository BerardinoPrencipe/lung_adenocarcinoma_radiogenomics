import os
import numpy as np
import SimpleITK as sitk
import vtk.util.numpy_support as VN
import vtk

#%%
def get_color_for_filename(name):
    color = (0.0,0.0,0.0)
    if 'venoussystem' in name or 'venacava' in name:
        color = (0.2,0.2,0.8)
    elif 'liver.vtk' in name:
        color = (0.8,0.2,0.6)
    elif 'portalvein' in name:
        color = (0.2,0.8,0.2)
    return color

#%%
base_folder = "F:/Datasets/3Dircadb1/3Dircadb1.1/MESHES_VTK"
# filename = os.path.join(base_folder, "venoussystem.vtk")
filenames = [os.path.join(base_folder, name) for name in os.listdir(base_folder)]
filenames_filt = [name for name in filenames if
                  'venoussystem' in name or 'venacava' in name or
                  'liver.vtk' in name or
                  'portalvein' in name]

# Setup render window, renderer, and interactor
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

for filename in filenames_filt:

    color = get_color_for_filename(filename)
    r,g,b = color

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    polydata = reader.GetOutput()

    # Setup actor and mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.GetProperty().GetPointSize()
    actor.GetProperty().SetColor(r,g,b)

    renderer.AddActor(actor)

renderer.SetBackground(.1,.1,.1)

renderWindow.Render()
renderWindowInteractor.Start()



