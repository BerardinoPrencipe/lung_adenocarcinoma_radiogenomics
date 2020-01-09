import vtk
import os

def get_color_for_filename(name):
    color = (0.0,0.0,0.0)
    if 'hv' in name:
        color = (0.2,0.2,0.8)
    elif 'liver' in name:
        color = (0.8,0.2,0.6)
    elif 'pv' in name:
        color = (0.2,0.8,0.2)
    elif 'artery' in name:
        color = (0.8,0.2,0.2)
    return color

dir_image = 'datasets/ircadb/patient-01/mask'
# path_image = os.path.join(dir_image, 'mask.nii')
path_images = [os.path.join(dir_image, name) for name in os.listdir(dir_image) if name != 'mask.nii']

# n_iters = 5000
n_iters = 100

renderWindow = vtk.vtkRenderWindow()
renderer = vtk.vtkRenderer()

for path_image in path_images:
    color = get_color_for_filename(path_image)
    r,g,b = color

    imageReader = vtk.vtkNIFTIImageReader()
    imageReader.SetFileName(path_image)

    dmc = vtk.vtkDiscreteMarchingCubes()
    # dmc = vtk.vtkMarchingCubes()
    dmc.SetInputConnection(imageReader.GetOutputPort())
    dmc.GenerateValues(1,1,1)
    dmc.Update()

    spdf = vtk.vtkSmoothPolyDataFilter()
    spdf.SetInputConnection(dmc.GetOutputPort())
    spdf.SetNumberOfIterations(n_iters)

    mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputConnection(dmc.GetOutputPort())
    mapper.SetInputConnection(spdf.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(r,g,b)
    actor.GetMapper().ScalarVisibilityOff()

    renderer.AddActor(actor)
    renderer.SetBackground(0.1,0.1,0.1)

renderWindow.AddRenderer(renderer)
renderWindow.Render()

renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderWindowInteractor.Start()