import pytest
from layout_prompter.models import (
    Coordinates,
    LayoutSerializedData,
    LayoutSerializedOutputData,
    PosterLayoutSerializedData,
    PosterLayoutSerializedOutputData,
    Rico25SerializedData,
    Rico25SerializedOutputData,
)
from pydantic import ValidationError


def test_coordinates():
    """Coordinatesクラスのテスト"""
    # インスタンス作成のテスト
    coords = Coordinates(left=10, top=20, width=30, height=40)

    # 属性が正しく設定されていることを確認
    assert coords.left == 10
    assert coords.top == 20
    assert coords.width == 30
    assert coords.height == 40

    # to_tupleメソッドのテスト
    assert coords.to_tuple() == (10, 20, 30, 40)


def test_coordinates_edge_cases():
    """Coordinatesクラスのエッジケーステスト"""
    # 負の値でのテスト
    negative_coords = Coordinates(left=-10, top=-20, width=30, height=40)
    assert negative_coords.left == -10
    assert negative_coords.top == -20
    assert negative_coords.to_tuple() == (-10, -20, 30, 40)

    # ゼロ値でのテスト
    zero_coords = Coordinates(left=0, top=0, width=0, height=0)
    assert zero_coords.left == 0
    assert zero_coords.top == 0
    assert zero_coords.width == 0
    assert zero_coords.height == 0
    assert zero_coords.to_tuple() == (0, 0, 0, 0)

    # 大きな値でのテスト
    large_coords = Coordinates(left=10000, top=20000, width=30000, height=40000)
    assert large_coords.left == 10000
    assert large_coords.top == 20000
    assert large_coords.width == 30000
    assert large_coords.height == 40000
    assert large_coords.to_tuple() == (10000, 20000, 30000, 40000)


def test_poster_layout_serialized_data():
    """PosterLayoutSerializedDataクラスのテスト"""
    # インスタンス作成のテスト
    serialized_data = PosterLayoutSerializedData(
        class_name="text", coord=Coordinates(left=10, top=20, width=30, height=40)
    )

    # 属性が正しく設定されていることを確認
    assert serialized_data.class_name == "text"
    assert serialized_data.coord.left == 10
    assert serialized_data.coord.top == 20
    assert serialized_data.coord.width == 30
    assert serialized_data.coord.height == 40

    # 異なるクラス名でのテスト
    serialized_data2 = PosterLayoutSerializedData(
        class_name="logo", coord=Coordinates(left=5, top=15, width=25, height=35)
    )
    assert serialized_data2.class_name == "logo"

    # 有効なクラス名のリスト
    valid_class_names = ["text", "logo", "underlay"]
    assert serialized_data.class_name in valid_class_names
    assert serialized_data2.class_name in valid_class_names


def test_poster_layout_serialized_data_invalid_class_name():
    """PosterLayoutSerializedDataで無効なクラス名を使用した場合のエラーテスト"""
    # 無効なクラス名でのインスタンス作成
    # 注: 型チェックのため、直接"invalid"を渡すとエラーになるので、
    # 変数に代入してから使用することで型チェックをバイパスする
    invalid_class = "invalid"
    with pytest.raises(ValidationError) as excinfo:
        PosterLayoutSerializedData(
            class_name=invalid_class,  # type: ignore
            coord=Coordinates(left=10, top=20, width=30, height=40),
        )

    # エラーメッセージに期待される文字列が含まれていることを確認
    assert "Input should be" in str(excinfo.value)


def test_poster_layout_serialized_output_data():
    """PosterLayoutSerializedOutputDataクラスのテスト"""
    # テスト用のPosterLayoutSerializedDataインスタンスを作成
    data1 = PosterLayoutSerializedData(
        class_name="text", coord=Coordinates(left=10, top=20, width=30, height=40)
    )
    data2 = PosterLayoutSerializedData(
        class_name="logo", coord=Coordinates(left=50, top=60, width=70, height=80)
    )

    # PosterLayoutSerializedOutputDataインスタンスの作成
    output_data = PosterLayoutSerializedOutputData(layouts=[data1, data2])

    # 属性が正しく設定されていることを確認
    assert len(output_data.layouts) == 2
    assert output_data.layouts[0].class_name == "text"
    assert output_data.layouts[1].class_name == "logo"
    assert output_data.layouts[0].coord.to_tuple() == (10, 20, 30, 40)
    assert output_data.layouts[1].coord.to_tuple() == (50, 60, 70, 80)


def test_poster_layout_serialized_output_data_edge_cases():
    """PosterLayoutSerializedOutputDataクラスのエッジケーステスト"""
    # 空のレイアウトリストでのテスト
    empty_output_data = PosterLayoutSerializedOutputData(layouts=[])
    assert len(empty_output_data.layouts) == 0

    # 多数のレイアウトでのテスト
    many_layouts = []
    for i in range(100):
        many_layouts.append(
            PosterLayoutSerializedData(
                class_name="text"
                if i % 3 == 0
                else "logo"
                if i % 3 == 1
                else "underlay",
                coord=Coordinates(left=i, top=i + 10, width=30, height=40),
            )
        )

    large_output_data = PosterLayoutSerializedOutputData(layouts=many_layouts)
    assert len(large_output_data.layouts) == 100
    assert large_output_data.layouts[0].class_name == "text"
    assert large_output_data.layouts[1].class_name == "logo"
    assert large_output_data.layouts[2].class_name == "underlay"


def test_rico25_serialized_data():
    """Rico25SerializedDataクラスのテスト"""
    # インスタンス作成のテスト
    serialized_data = Rico25SerializedData(
        class_name="text", coord=Coordinates(left=10, top=20, width=30, height=40)
    )

    # 属性が正しく設定されていることを確認
    assert serialized_data.class_name == "text"
    assert serialized_data.coord.left == 10
    assert serialized_data.coord.top == 20
    assert serialized_data.coord.width == 30
    assert serialized_data.coord.height == 40

    # 異なるクラス名でのテスト
    serialized_data2 = Rico25SerializedData(
        class_name="image", coord=Coordinates(left=5, top=15, width=25, height=35)
    )
    assert serialized_data2.class_name == "image"


def test_rico25_serialized_data_all_valid_classes():
    """Rico25SerializedDataの全ての有効なクラス名のテスト"""
    valid_classes = [
        "text",
        "image",
        "icon",
        "list-item",
        "text-button",
        "toolbar",
        "web-view",
        "input",
        "card",
        "advertisement",
        "background-image",
        "drawer",
        "radio-button",
        "checkbox",
        "multi-tab",
        "pager-indicator",
        "modal",
        "on/off-switch",
        "slider",
        "map-view",
        "button-bar",
        "video",
        "bottom-navigation",
        "number-stepper",
        "date-picker",
    ]

    for class_name in valid_classes:
        data = Rico25SerializedData(
            class_name=class_name,  # type: ignore
            coord=Coordinates(left=10, top=20, width=30, height=40),
        )
        assert data.class_name == class_name


def test_rico25_serialized_data_invalid_class_name():
    """Rico25SerializedDataで無効なクラス名を使用した場合のエラーテスト"""
    # 無効なクラス名でのインスタンス作成
    # 注: 型チェックのため、直接"invalid"を渡すとエラーになるので、
    # 変数に代入してから使用することで型チェックをバイパスする
    invalid_class = "non_existent_class"
    with pytest.raises(ValidationError) as excinfo:
        Rico25SerializedData(
            class_name=invalid_class,  # type: ignore
            coord=Coordinates(left=10, top=20, width=30, height=40),
        )

    # エラーメッセージに期待される文字列が含まれていることを確認
    assert "Input should be" in str(excinfo.value)


def test_rico25_serialized_output_data():
    """Rico25SerializedOutputDataクラスのテスト"""
    # テスト用のRico25SerializedDataインスタンスを作成
    data1 = Rico25SerializedData(
        class_name="text", coord=Coordinates(left=10, top=20, width=30, height=40)
    )
    data2 = Rico25SerializedData(
        class_name="image", coord=Coordinates(left=50, top=60, width=70, height=80)
    )

    # Rico25SerializedOutputDataインスタンスの作成
    output_data = Rico25SerializedOutputData(layouts=[data1, data2])

    # 属性が正しく設定されていることを確認
    assert len(output_data.layouts) == 2
    assert output_data.layouts[0].class_name == "text"
    assert output_data.layouts[1].class_name == "image"
    assert output_data.layouts[0].coord.to_tuple() == (10, 20, 30, 40)
    assert output_data.layouts[1].coord.to_tuple() == (50, 60, 70, 80)


def test_rico25_serialized_output_data_edge_cases():
    """Rico25SerializedOutputDataクラスのエッジケーステスト"""
    # 空のレイアウトリストでのテスト
    empty_output_data = Rico25SerializedOutputData(layouts=[])
    assert len(empty_output_data.layouts) == 0

    # 多数のレイアウトでのテスト
    many_layouts = []
    valid_classes = ["text", "image", "icon", "list-item", "text-button"]

    for i in range(100):
        many_layouts.append(
            Rico25SerializedData(
                class_name=valid_classes[i % len(valid_classes)],  # type: ignore
                coord=Coordinates(left=i, top=i + 10, width=30, height=40),
            )
        )

    large_output_data = Rico25SerializedOutputData(layouts=many_layouts)
    assert len(large_output_data.layouts) == 100
    assert large_output_data.layouts[0].class_name == "text"
    assert large_output_data.layouts[1].class_name == "image"
    assert large_output_data.layouts[2].class_name == "icon"


def test_generic_serialized_data():
    """Generic SerializedDataクラスのテスト"""
    # 基本的なインスタンス作成のテスト
    generic_data = LayoutSerializedData(
        class_name="test_class", coord=Coordinates(left=10, top=20, width=30, height=40)
    )

    # 属性が正しく設定されていることを確認
    assert generic_data.class_name == "test_class"
    assert generic_data.coord.left == 10
    assert generic_data.coord.top == 20
    assert generic_data.coord.width == 30
    assert generic_data.coord.height == 40


def test_generic_serialized_output_data():
    """Generic SerializedOutputDataクラスのテスト"""
    # テスト用のSerializedDataインスタンスを作成
    data1 = LayoutSerializedData(
        class_name="class1", coord=Coordinates(left=10, top=20, width=30, height=40)
    )
    data2 = LayoutSerializedData(
        class_name="class2", coord=Coordinates(left=50, top=60, width=70, height=80)
    )

    # SerializedOutputDataインスタンスの作成
    output_data = LayoutSerializedOutputData(layouts=[data1, data2])

    # 属性が正しく設定されていることを確認
    assert len(output_data.layouts) == 2
    assert output_data.layouts[0].class_name == "class1"
    assert output_data.layouts[1].class_name == "class2"
    assert output_data.layouts[0].coord.to_tuple() == (10, 20, 30, 40)
    assert output_data.layouts[1].coord.to_tuple() == (50, 60, 70, 80)


def test_type_compatibility():
    """型の互換性テスト"""
    # 具体的なインスタンスを作成
    poster_data = PosterLayoutSerializedData(
        class_name="text", coord=Coordinates(left=10, top=20, width=30, height=40)
    )
    rico_data = Rico25SerializedData(
        class_name="image", coord=Coordinates(left=50, top=60, width=70, height=80)
    )

    # 基底型として参照できることを確認
    base_poster: LayoutSerializedData = poster_data
    base_rico: LayoutSerializedData = rico_data

    assert base_poster.class_name == "text"
    assert base_rico.class_name == "image"

    # Generic の出力データでも互換性があることを確認
    mixed_output = LayoutSerializedOutputData(layouts=[poster_data, rico_data])
    assert len(mixed_output.layouts) == 2
    assert mixed_output.layouts[0].class_name == "text"
    assert mixed_output.layouts[1].class_name == "image"


def test_inheritance_structure():
    """継承構造のテスト"""
    # 継承関係の確認
    poster_data = PosterLayoutSerializedData(
        class_name="text", coord=Coordinates(left=10, top=20, width=30, height=40)
    )
    rico_data = Rico25SerializedData(
        class_name="image", coord=Coordinates(left=50, top=60, width=70, height=80)
    )

    # isinstance での型チェック
    assert isinstance(poster_data, LayoutSerializedData)
    assert isinstance(rico_data, LayoutSerializedData)
    assert isinstance(poster_data, PosterLayoutSerializedData)
    assert isinstance(rico_data, Rico25SerializedData)

    # 型の特定
    assert type(poster_data).__name__ == "PosterLayoutSerializedData"
    assert type(rico_data).__name__ == "Rico25SerializedData"


def test_class_name_validation():
    """クラス名のバリデーションテスト"""
    # LayoutSerializedData は任意の文字列を受け入れる
    generic_data = LayoutSerializedData(
        class_name="any_string", coord=Coordinates(left=10, top=20, width=30, height=40)
    )
    assert generic_data.class_name == "any_string"

    # PosterLayoutSerializedData は特定のクラス名のみ受け入れる
    valid_poster_names = ["text", "logo", "underlay"]
    for name in valid_poster_names:
        data = PosterLayoutSerializedData(
            class_name=name, coord=Coordinates(left=10, top=20, width=30, height=40)
        )
        assert data.class_name == name

    # Rico25SerializedData は特定のクラス名のみ受け入れる
    valid_rico_names = ["text", "image", "icon"]
    for name in valid_rico_names:
        data = Rico25SerializedData(
            class_name=name, coord=Coordinates(left=10, top=20, width=30, height=40)
        )
        assert data.class_name == name


def test_output_data_flexibility():
    """出力データの柔軟性テスト"""
    # 異なる種類のSerializedDataを混在させられることを確認
    poster_data = PosterLayoutSerializedData(
        class_name="text", coord=Coordinates(left=10, top=20, width=30, height=40)
    )
    rico_data = Rico25SerializedData(
        class_name="image", coord=Coordinates(left=50, top=60, width=70, height=80)
    )
    generic_data = LayoutSerializedData(
        class_name="custom", coord=Coordinates(left=100, top=120, width=130, height=140)
    )

    # LayoutSerializedOutputData で混在させる
    mixed_output = LayoutSerializedOutputData(
        layouts=[poster_data, rico_data, generic_data]
    )
    assert len(mixed_output.layouts) == 3
    assert mixed_output.layouts[0].class_name == "text"
    assert mixed_output.layouts[1].class_name == "image"
    assert mixed_output.layouts[2].class_name == "custom"

    # 専用の出力データクラスでは同じ種類のみ
    poster_output = PosterLayoutSerializedOutputData(layouts=[poster_data])
    assert len(poster_output.layouts) == 1
    assert poster_output.layouts[0].class_name == "text"

    rico_output = Rico25SerializedOutputData(layouts=[rico_data])
    assert len(rico_output.layouts) == 1
    assert rico_output.layouts[0].class_name == "image"


def test_backward_compatibility():
    """後方互換性テスト"""
    # 既存のコードが引き続き動作することを確認
    coord = Coordinates(left=10, top=20, width=30, height=40)

    # 基本的なSerializedDataの使用
    data = LayoutSerializedData(class_name="test", coord=coord)
    assert data.class_name == "test"
    assert data.coord.to_tuple() == (10, 20, 30, 40)

    # SerializedOutputDataの使用
    output = LayoutSerializedOutputData(layouts=[data])
    assert len(output.layouts) == 1
    assert output.layouts[0].class_name == "test"
