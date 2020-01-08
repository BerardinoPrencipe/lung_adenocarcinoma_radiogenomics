import vtk

# dir_mask = 'datasets/ircadb/patient-01/mask/'
# filename_mask = os.path.join(dir_mask, 'mask.dcm')
dir_mask = 'F:/Datasets/LiverScardapane/BENNARD/1/7'

imageReader = vtk.vtkDICOMImageReader()
# imageReader.SetFileName(filename_mask)
imageReader.SetDirectoryName(dir_mask)
imageReader.Update()

dmc = vtk.vtkDiscreteMarchingCubes()
dmc.SetInputConnection(imageReader.GetOutputPort())
dmc.GenerateValues(1,1,1)
dmc.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(dmc.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1.0,1.0,1.0)

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.Render()

renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderWindowInteractor.Start()