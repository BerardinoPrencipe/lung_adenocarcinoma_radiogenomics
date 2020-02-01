import vtk
import os

do_smoothing = True
n_iters = 1500

r,g,b = tuple(i/255 for i in (171,23,65))

''' 
path_image_pred_rg  = 'E:/Datasets/Sliver_Nifti/Results/RegionGrowing/D25/liver-seg020.nii'
path_image_pred_cnn = 'E:/Datasets/Sliver_Nifti/Results/CNN/a5_b5/post/segmentation-orig020.nii'
path_image_gt       = 'E:/Datasets/Sliver_Nifti/GroundTruth/liver-seg020.nii.gz'

path_mesh_pred_rg  = 'datasets/Sliver07/liver_rg_20.wrl'
path_mesh_pred_cnn = 'datasets/Sliver07/liver_cnn_20.wrl'
path_mesh_gt       = 'datasets/Sliver07/liver_gt_20.wrl' 
'''

idxs = range(1,5)

for idx in idxs:

    print('Index {} on {}'.format(idx, len(idxs)))

    path_image_pred  = 'datasets/ircadb/patient-{:02d}/image/pred.nii'.format(idx)
    path_image_hv_gt = 'datasets/ircadb/patient-{:02d}/mask/hv.nii'.format(idx)
    path_image_pv_gt = 'datasets/ircadb/patient-{:02d}/mask/pv.nii'.format(idx)

    mesh_dir = 'datasets/ircadb/patient-{:02d}/mesh/'.format(idx)
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
        print('Created mesh dir = {}'.format(mesh_dir))

    path_mesh_pred   = os.path.join(mesh_dir, 'pred.wrl')
    path_mesh_hv_gt  = os.path.join(mesh_dir, 'mask_gt_hv.wrl')
    path_mesh_pv_gt  = os.path.join(mesh_dir, 'mask_gt_pv.wrl')

    path_images = list((path_image_pred, path_image_hv_gt, path_image_pv_gt))
    path_meshes = list((path_mesh_pred, path_mesh_hv_gt, path_mesh_pv_gt))

    for path_image, path_mesh in zip(path_images, path_meshes):

        print('Path Image = {}\nPath Mesh = {}'.format(path_image, path_mesh))

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
        exporter.SetFileName(path_mesh)
        exporter.Write()
        exporter.Update()

        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderWindowInteractor.Start()
