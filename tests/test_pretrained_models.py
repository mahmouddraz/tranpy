"""
Comprehensive test suite for pretrained model registry and loading.
"""

import pytest
import json
from pathlib import Path
from tranpy.models import list_pretrained_models


class TestPretrainedModelRegistry:
    """Test pretrained model registry structure and content."""

    def test_registry_exists(self):
        """Test that registry file exists."""
        from tranpy.models.pretrained import _load_registry

        registry = _load_registry()
        assert registry is not None
        assert 'models' in registry
        assert 'version' in registry

    def test_registry_structure(self):
        """Test registry has correct structure."""
        from tranpy.models.pretrained import _load_registry

        registry = _load_registry()

        assert 'newengland' in registry['models']
        assert 'ninebussystem' in registry['models']

    def test_list_all_pretrained_models(self):
        """Test listing all pretrained models."""
        models = list_pretrained_models()

        assert isinstance(models, list)
        assert len(models) > 0

        # Check each model has required fields
        for model in models:
            assert 'model_id' in model
            assert 'type' in model
            assert 'grid' in model
            assert 'size_kb' in model
            assert 'description' in model
            assert 'google_drive_folder' in model
            assert 'filename' in model
            assert 'cached' in model

    def test_list_newengland_models(self):
        """Test filtering models by New England grid."""
        models = list_pretrained_models(grid='NewEngland')

        assert len(models) > 0
        for model in models:
            assert model['grid'] == 'NewEngland'

    def test_list_ninebus_models(self):
        """Test filtering models by Nine Bus grid."""
        models = list_pretrained_models(grid='NineBusSystem')

        assert len(models) > 0
        for model in models:
            assert model['grid'] == 'NineBusSystem'

    def test_model_types_coverage(self):
        """Test that registry includes various model types."""
        models = list_pretrained_models()
        model_types = {m['type'] for m in models}

        # Should have multiple model types
        assert len(model_types) >= 3

        # Check for expected types
        expected_types = {
            'AdaBoostClassifier',
            'DecisionTreeClassifier',
            'ExtraTreesClassifier',
            'GaussianNB',
            'DummyClassifier'
        }

        # At least some expected types should be present
        assert len(expected_types & model_types) >= 3

    def test_both_grids_have_models(self):
        """Test that both grid systems have models."""
        ne_models = list_pretrained_models(grid='NewEngland')
        nb_models = list_pretrained_models(grid='NineBusSystem')

        assert len(ne_models) > 0, "NewEngland should have models"
        assert len(nb_models) > 0, "NineBusSystem should have models"

    def test_model_ids_are_unique(self):
        """Test that all model IDs are unique."""
        models = list_pretrained_models()
        model_ids = [m['model_id'] for m in models]

        assert len(model_ids) == len(set(model_ids)), "Model IDs must be unique"

    def test_google_drive_links_present(self):
        """Test that all models have Google Drive information."""
        models = list_pretrained_models()

        for model in models:
            assert 'google_drive_folder' in model
            assert model['google_drive_folder'].startswith('https://')
            assert 'drive.google.com' in model['google_drive_folder']

    def test_filenames_match_convention(self):
        """Test that filenames follow naming convention."""
        models = list_pretrained_models()

        for model in models:
            filename = model['filename']
            assert filename.endswith('.pkl'), f"Filename should be .pkl: {filename}"
            assert model['grid'] in filename or 'NineBus' in filename, \
                f"Filename should contain grid name: {filename}"


class TestPretrainedModelLoading:
    """Test pretrained model loading functionality."""

    def test_load_pretrained_invalid_id(self):
        """Test that loading invalid model ID raises error."""
        from tranpy.models import load_pretrained

        with pytest.raises(ValueError, match="Unknown model ID"):
            load_pretrained('nonexistent_model_xyz123')

    def test_load_pretrained_without_download(self):
        """Test that loading requires download or manual setup."""
        from tranpy.models import load_pretrained

        # Get first model from registry
        models = list_pretrained_models()
        if models:
            model_id = models[0]['model_id']

            # If model is not cached, should raise NotImplementedError
            # (because file_id is not configured for download)
            try:
                model = load_pretrained(model_id)
                # If it loads, verify it's a valid model
                assert hasattr(model, 'predict') or hasattr(model, 'model_')
            except NotImplementedError as e:
                # Expected if file_id not configured
                assert 'file_id' in str(e) or 'file ID' in str(e)
            except FileNotFoundError:
                # Expected if not cached and can't download
                pass


class TestModelRegistryConsistency:
    """Test consistency between registry and expected models."""

    def test_newengland_has_expected_models(self):
        """Test New England has expected model types."""
        models = list_pretrained_models(grid='NewEngland')
        model_types = {m['type'] for m in models}

        # Should have at least these key models
        expected = {'DecisionTreeClassifier', 'AdaBoostClassifier'}
        missing = expected - model_types

        assert len(missing) == 0, f"Missing models for NewEngland: {missing}"

    def test_ninebus_has_expected_models(self):
        """Test Nine Bus has expected model types."""
        models = list_pretrained_models(grid='NineBusSystem')
        model_types = {m['type'] for m in models}

        # Should have at least these key models
        expected = {'DecisionTreeClassifier', 'AdaBoostClassifier'}
        missing = expected - model_types

        assert len(missing) == 0, f"Missing models for NineBusSystem: {missing}"

    def test_model_count_parity(self):
        """Test that both grids have similar number of models."""
        ne_models = list_pretrained_models(grid='NewEngland')
        nb_models = list_pretrained_models(grid='NineBusSystem')

        # Both should have models
        assert len(ne_models) > 0
        assert len(nb_models) > 0

        # Should have roughly the same number (within 2)
        diff = abs(len(ne_models) - len(nb_models))
        assert diff <= 2, f"Model count mismatch: NE={len(ne_models)}, NB={len(nb_models)}"

    def test_registry_json_is_valid(self):
        """Test that registry JSON file is valid."""
        registry_path = Path(__file__).parent.parent / 'src' / 'tranpy' / 'data' / 'models' / 'registry.json'

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        # Verify structure
        assert 'models' in registry
        assert 'version' in registry
        assert 'description' in registry

        # Verify models structure
        for grid_name, grid_models in registry['models'].items():
            assert isinstance(grid_models, dict)
            for model_key, model_info in grid_models.items():
                # Required fields
                assert 'model_id' in model_info
                assert 'model_type' in model_info
                assert 'grid' in model_info
                assert 'filename' in model_info
                assert 'google_drive' in model_info
                assert 'size_kb' in model_info

                # Google Drive info
                gd = model_info['google_drive']
                assert 'folder_id' in gd
                assert 'folder_link' in gd


class TestModelNamingConventions:
    """Test model naming conventions and IDs."""

    def test_model_id_format(self):
        """Test that model IDs follow convention."""
        models = list_pretrained_models()

        for model in models:
            model_id = model['model_id']

            # Should contain grid indicator
            assert '_ne39' in model_id or '_9bus' in model_id, \
                f"Model ID should indicate grid: {model_id}"

            # Should be lowercase with underscores
            assert model_id == model_id.lower(), \
                f"Model ID should be lowercase: {model_id}"
            assert ' ' not in model_id, \
                f"Model ID should not contain spaces: {model_id}"

    def test_filename_consistency(self):
        """Test filename matches grid and type."""
        models = list_pretrained_models()

        for model in models:
            filename = model['filename']
            grid = model['grid']
            model_type = model['type']

            # Filename should contain grid name
            assert grid in filename or ('NineBus' in filename and grid == 'NineBusSystem'), \
                f"Filename should contain grid: {filename}"

            # Filename should contain model type
            assert model_type in filename, \
                f"Filename should contain type {model_type}: {filename}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
