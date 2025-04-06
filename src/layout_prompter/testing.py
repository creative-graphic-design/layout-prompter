import pathlib


class LayoutPrompterTestCase(object):
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
    MODULE_ROOT = PROJECT_ROOT / "layout_prompter"
    TEST_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"
