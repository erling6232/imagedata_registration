# Changelog / release notes

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--next-version-placeholder-->

## [v0.3.2] - 2024-12-11
* Minor improvements.
* Run build script on latest ubuntu and macos os versions.

## [v0.3.1] - 2024-10-10
* Support python 3.13.

## [v0.3.0] - 2024-10-09
* SimpleElastix has been discontinued.
* Reworked the ITK Elastix part to work ITK-Elastix from the Insight Toolkit (ITK).

## [v0.2.7] - 2024-10-08
* Require imagedata 3.7.

## [v0.2.6] - 2024-06-27
* Verify function with imagedata==3.6.0.
* Drop support for python 3.8.

## [v0.2.4] - 2024-03-11
### Fixed
* Modified FSL behaviour to work with updated imagedata 3.5.0 saving NIfTI files
  with explicit file names.
* Accept python 3.12.

## [v0.2.3] - 2023-12-15
### Fixed
* Corrected handling of multigrid.

### Added
* Documented the use of Series objects with SimpleElastix methods.

## [v0.2.2] - 2023-12-01
### Added
* Added Elastix.register_elastix_parametermap() to take an Elastix parameter map
  to configure the registration components.
* Updated documentation on registering with Elastix.
