def test_import_tranpy():
    import tranpy
    assert hasattr(tranpy, "__file__") or hasattr(tranpy, "__doc__")
