import pytest
from pydantic import ValidationError

from layout_prompter.models import (
    Coordinates,
    PosterLayoutSerializedData,
    PosterLayoutSerializedOutputData,
    Rico25SerializedData,
    Rico25SerializedOutputData,
)


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
