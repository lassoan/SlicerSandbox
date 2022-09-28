import logging
import os

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


#
# NgffImageIO
#

class NgffImageIO(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "NgffImageIO"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#NgffImageIO">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # NgffImageIO1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='NgffImageIO',
        sampleName='NgffImageIO1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'NgffImageIO1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='NgffImageIO1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='NgffImageIO1'
    )

    # NgffImageIO2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='NgffImageIO',
        sampleName='NgffImageIO2',
        thumbnailFileName=os.path.join(iconsPath, 'NgffImageIO2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='NgffImageIO2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='NgffImageIO2'
    )

#
# NgffImageIOLogic
#

class NgffImageIOLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def xarrayFromVolume(volumeNode) -> "xr.DataArray":
        """Convert a volume node to an xarray.DataArray.
        Since image axes may be rotated in physical space but xarray accessors do not
        support rotated axes, the image in the xarray is defined in the "xyz" space,
        which is voxel space scaled with the image spacing.
        `xyzToPhysicalTransform` attribute stores a 4x4 homogeneous transformation matrix
        to transform between xyz to physical (LPS) coordinate systems.
        Spacing metadata is preserved in the xarray's coords.
        Dims are labeled as `x`, `y`, `z`, `t`, and `c`.
        This interface is and behavior is experimental and is subject to possible
        future changes."""
        import xarray as xr
        import numpy as np
        array_view = slicer.util.arrayFromVolume(volumeNode)
        spacing = volumeNode.GetSpacing()
        origin_ras = volumeNode.GetOrigin()
        origin_lps = [-origin_ras[0], -origin_ras[1], origin_ras[2]]
        size = volumeNode.GetImageData().GetDimensions()
        image_dimension = 3
        image_dims = ("x", "y", "z", "t")
        coords = {}
        # When we export an image, xyz origin is always set to (0,0,0), but after processing
        # (such as resampling or extracting a subset of the data), the origin may change.
        origin_xyz = [0.0, 0.0, 0.0]
        for index, dim in enumerate(image_dims[:image_dimension]):
            coords[dim] = np.linspace(
                origin_xyz[index],
                origin_xyz[index] + (size[index] - 1) * spacing[index],
                size[index],
                dtype=np.float64,
            )
        dims = list(reversed(image_dims[:image_dimension]))
        components = volumeNode.GetImageData().GetNumberOfScalarComponents()
        if components > 1:
            dims.append("c")
            coords["c"] = np.arange(components, dtype=np.uint32)
        ijkToRasMatrixVtk = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRasMatrixVtk)
        ijkToRasMatrix = slicer.util.arrayFromVTKMatrix(ijkToRasMatrixVtk)
        ijkToLpsMatrix = np.dot(ijkToRasMatrix, np.diag([-1.0, -1.0, 1.0, 1.0]))
        xyzToIjkMatrix = np.diag([1.0/spacing[0], 1.0/spacing[1], 1.0/spacing[2], 1.0])
        xyzToLpsMatrix = np.dot(ijkToLpsMatrix, xyzToIjkMatrix)
        print(f"xyzToPhysical={xyzToLpsMatrix}")
        attrs = {"xyzToPhysicalTransform": np.flip(xyzToLpsMatrix)}
        for attributeName in volumeNode.GetAttributeNames():
            attrs[key] = volumeNode.GetAttribute(attributeName)
        data_array = xr.DataArray(array_view, dims=dims, coords=coords, attrs=attrs)
        return data_array

    def volumeFromXarray(data_array: "xr.DataArray"):
        """Convert an xarray.DataArray to a MRML volume node.
        """
        import numpy as np
        import builtins
        if not {"t", "z", "y", "x", "c"}.issuperset(data_array.dims):
            raise ValueError('Unsupported dims, supported dims: "t", "z", "y", "x", "c".')
        image_dims = list({"t", "z", "y", "x"}.intersection(set(data_array.dims)))
        image_dims.sort(reverse=True)
        image_dimension = len(image_dims)
        ordered_dims = ("t", "z", "y", "x")[-image_dimension:]
        is_vector = "c" in data_array.dims
        if is_vector:
            ordered_dims = ordered_dims + ("c",)
        values = data_array.values
        if ordered_dims != data_array.dims:
            dest = list(builtins.range(len(ordered_dims)))
            source = dest.copy()
            for ii in builtins.range(len(ordered_dims)):
                source[ii] = data_array.dims.index(ordered_dims[ii])
            values = np.moveaxis(values, source, dest).copy()
        volumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode' if not is_vector else 'vtkMRMLVectorVolumeNode')
        slicer.util.updateVolumeFromArray(volumeNode, values) # is_vector)
        origin_xyz = [0.0] * image_dimension
        spacing = [1.0] * image_dimension
        for index, dim in enumerate(image_dims):
            coords = data_array.coords[dim]
            if coords.shape[0] > 1:
                origin_xyz[index] = float(coords[0])
                print(f'origin[{dim}] = {origin_xyz[index]}')
                spacing[index] = float(coords[-1] - coords[0]) / float(len(coords)-1)
                print(f'coords[{dim}] spacing = ({coords[-1]} - {coords[0]}) / {len(coords)-1} = {spacing[index]}')
        spacing.reverse()
        origin_xyz.reverse()
        ijkToXyz = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
        ijkToXyz[:,3] = [-origin_xyz[0], -origin_xyz[1], origin_xyz[2], 1.0]  # TODO: it is not clear why first two components need sign inversion
        print(f"ijkToXyz={ijkToXyz}")
        if "xyzToPhysicalTransform" in data_array.attrs:
            xyzToPhysical = np.flip(data_array.attrs["xyzToPhysicalTransform"])
        else:
            xyzToPhysical = np.identity(4)
        ijkToLps = np.dot(xyzToPhysical, ijkToXyz)
        ijkToRas = np.dot(ijkToLps, np.diag([-1.0, -1.0, 1.0, 1.0]))
        print(f"xyzToPhysical={xyzToPhysical}")
        print(f"ijkToRas={ijkToRas}")
        volumeNode.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(ijkToRas))
        ignore_keys = set(["xyzToPhysicalTransform"])
        for key in data_array.attrs:
            if not key in ignore_keys:
                volumeNode.SetAttribute(key, data_array.attrs[key])
        return volumeNode


#
# NgffImageIOTest
#

class NgffImageIOTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_NgffImageIO1()

    def test_NgffImageIO1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('NgffImageIO1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = NgffImageIOLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')

        # Convert to xarray
        import SampleData
        sampleDataLogic = SampleData.SampleDataLogic()
        #volumeNode = sampleDataLogic.downloadMRBrainTumor1() 
        volumeNode = sampleDataLogic.downloadMRHead()
        da = xarrayFromVolume(volumeNode)
        print(da)

        # Save as zarr
        ds = da.to_dataset(name='image').chunk({'x': 20, 'y': -1})
        zs = ds.to_zarr('c:/tmp/zarrimage/', mode='w')

        # Load from zarr
        import xarray as xr
        ds=xr.open_dataset(r'c:\tmp\zarrimage', engine='zarr')

        # Convert to volume
        volumeNode = volumeFromXarray(da)

        # Display volume
        setSliceViewerLayers(volumeNode)

        # Load region of a volume
        ds=xr.open_dataset(r'c:\tmp\zarrimage', engine='zarr', chunks={})

        import numpy as np
        dsPart = ds.sel(x=np.arange(50, 180, 2), y=np.arange(40, 160, 2), z=np.arange(50, 60, 2), method='nearest').image
        volumePart = volumeFromXarray(dsPart)
        volumePart.CreateDefaultDisplayNodes()
        volumePart.GetScalarVolumeDisplayNode().SetAndObserveColorNodeID('vtkMRMLColorTableNodeRed')
        setSliceViewerLayers(foreground=volumePart, foregroundOpacity=0.5)
