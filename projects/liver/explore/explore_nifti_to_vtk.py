import vtk
import os


path_image_pred_rg = 'E:/Datasets/Sliver_Nifti/Results/RegionGrowing/D25/liver-seg020.nii'
path_image_pred_cnn = 'E:/Datasets/Sliver_Nifti/Results/CNN/a5_b5/post/segmentation-orig020.nii'
path_image_gt  = 'E:/Datasets/Sliver_Nifti/GroundTruth/liver-seg020.nii.gz'

# path_image = path_image_gt
# path_image = path_image_pred_cnn
path_image = path_image_pred_rg

do_smoothing = True
# n_iters = 5000
n_iters = 1500

r,g,b = tuple(i/255 for i in (171,23,65))

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
if do_smoothing:
    mapper.SetInputConnection(spdf.GetOutputPort())
else:
    mapper.SetInputConnection(dmc.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(r,g,b)
actor.GetMapper().ScalarVisibilityOff()

renderWindow = vtk.vtkRenderWindow()
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

renderer.SetBackground(0.1,0.1,0.1)
renderWindow.AddRenderer(renderer)
renderWindow.Render()

exporter = vtk.vtkVRMLExporter()
exporter.SetRenderWindow(renderWindow)
exporter.SetFileName('datasets/Sliver07/liver_rg_20.wrl')
# exporter.SetFileName('datasets/Sliver07/liver_cnn_20.wrl')
# exporter.SetFileName('datasets/Sliver07/liver_gt_20.wrl')
exporter.Write()
exporter.Update()

renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderWindowInteractor.Start()
