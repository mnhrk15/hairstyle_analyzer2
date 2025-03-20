"""
テンプレート選択機能の統合テスト
"""

import unittest
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import os
import sys
import streamlit as st
import pandas as pd

# テスト対象のモジュールをインポート
from hairstyle_analyzer.ui.streamlit_app import (
    init_session_state,
    display_template_choices,
    display_results,
    SESSION_RESULTS,
    SESSION_TEMPLATE_CHOICES,
    SESSION_USER_SELECTIONS,
    SESSION_PROCESSING_STAGES
)
from hairstyle_analyzer.data.models import (
    ProcessResult, 
    StyleAnalysis, 
    AttributeAnalysis, 
    Template, 
    StylistInfo, 
    CouponInfo, 
    StyleFeatures
)

@pytest.fixture
def mock_session_state():
    """セッション状態のモック"""
    mock_state = {
        SESSION_TEMPLATE_CHOICES: {},
        SESSION_USER_SELECTIONS: {},
        SESSION_PROCESSING_STAGES: {
            'processing_done': True,
            'template_selection_done': False,
            'results_display_done': False
        }
    }
    return mock_state

@pytest.fixture
def sample_template():
    """サンプルテンプレート"""
    return Template(
        category="ミディアムボブ",
        title="ふんわりナチュラルミディアムボブ",
        menu="カット+カラー",
        comment="柔らかな質感が魅力的なナチュラルスタイル",
        hashtag="#ナチュラル,#ミディアムボブ,#アッシュブラウン"
    )

@pytest.fixture
def sample_alternative_templates():
    """サンプル代替テンプレート"""
    return [
        Template(
            category="ミディアムボブ",
            title="大人かわいいウェーブボブ",
            menu="カット+パーマ",
            comment="ゆるふわパーマで動きのあるスタイル",
            hashtag="#ゆるふわ,#ウェーブ,#大人かわいい"
        ),
        Template(
            category="ミディアムレイヤー",
            title="シースルーバングミディアム",
            menu="カット+前髪カット",
            comment="軽やかな印象の前髪が特徴",
            hashtag="#シースルーバング,#ミディアム,#軽やか"
        )
    ]

@pytest.fixture
def sample_process_result(sample_template, sample_alternative_templates):
    """サンプル処理結果"""
    # スタイル分析の作成
    style_analysis = StyleAnalysis(
        category="ミディアムボブ",
        features=StyleFeatures(
            color="アッシュブラウン",
            cut_technique="グラデーションカット",
            styling="ナチュラルウェーブ",
            impression="柔らかな印象"
        ),
        keywords=["ナチュラル", "ウェーブ", "ボブ"]
    )
    
    # 属性分析の作成
    attribute_analysis = AttributeAnalysis(
        sex="レディース",
        length="ミディアム"
    )
    
    # スタイリスト情報の作成
    stylist = StylistInfo(
        name="山田優子",
        specialties="カット・カラーが得意",
        description="10年のキャリアを持つ実力派スタイリスト"
    )
    
    # クーポン情報の作成
    coupon = CouponInfo(
        name="平日限定カット+カラークーポン",
        price=10000,
        description="平日限定でお得なクーポン"
    )
    
    # 処理結果の作成
    return ProcessResult(
        image_name="test_image.jpg",
        style_analysis=style_analysis,
        attribute_analysis=attribute_analysis,
        selected_template=sample_template,
        alternative_templates=sample_alternative_templates,
        selected_stylist=stylist,
        selected_coupon=coupon
    )

class TestTemplateSelection:
    """テンプレート選択機能のテストクラス"""
    
    @patch('hairstyle_analyzer.ui.streamlit_app.get_config_manager')
    @patch('streamlit.expander')
    @patch('streamlit.columns')
    @patch('streamlit.radio')
    @patch('streamlit.button')
    def test_template_selection_workflow(self, mock_button, mock_radio, mock_columns, 
                                          mock_expander, mock_get_config, monkeypatch,
                                          mock_session_state, sample_process_result):
        """テンプレート選択ワークフローのテスト"""
        # セッション状態のセットアップ
        monkeypatch.setattr(st, "session_state", mock_session_state)
        
        # ConfigManagerのモック
        mock_config = MagicMock()
        mock_config.paths.image_folder = "assets/samples"
        mock_get_config.return_value = mock_config
        
        # エクスパンダーのモック
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_context
        
        # カラムのモック
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        # ラジオボタンのモック（1番目の代替テンプレートを選択）
        mock_radio.return_value = 1
        
        # ボタンのモック（確定ボタンを押す）
        mock_button.return_value = True
        
        # 処理結果のリスト
        results = [sample_process_result]
        
        # テンプレート選択関数の呼び出し
        display_template_choices(results)
        
        # セッション状態にテンプレート選択肢が保存されていることを確認
        assert "test_image.jpg" in st.session_state[SESSION_TEMPLATE_CHOICES]
        assert len(st.session_state[SESSION_TEMPLATE_CHOICES]["test_image.jpg"]) == 3
        
        # ユーザー選択が保存されていることを確認
        assert "test_image.jpg" in st.session_state[SESSION_USER_SELECTIONS]
        assert st.session_state[SESSION_USER_SELECTIONS]["test_image.jpg"] == 1
        
        # ProcessResultオブジェクトにユーザー選択が反映されていることを確認
        assert results[0].user_selected_template == sample_process_result.alternative_templates[0]
        
        # 確定ボタンが押されたフラグがセットされていることを確認
        assert st.session_state.get('confirm_template_selections', False) == True
    
    @patch('hairstyle_analyzer.ui.streamlit_app.get_config_manager')
    @patch('streamlit.dataframe')
    @patch('streamlit.expander')
    def test_results_display_with_user_selection(self, mock_expander, mock_dataframe, 
                                                  mock_get_config, monkeypatch,
                                                  mock_session_state, sample_process_result):
        """ユーザー選択を含む結果表示のテスト"""
        # セッション状態のセットアップ
        monkeypatch.setattr(st, "session_state", mock_session_state)
        
        # ConfigManagerのモック
        mock_config = MagicMock()
        mock_config.paths.image_folder = "assets/samples"
        mock_get_config.return_value = mock_config
        
        # エクスパンダーのモック
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_context
        
        # ユーザー選択をセット
        sample_process_result.user_selected_template = sample_process_result.alternative_templates[0]
        
        # 処理結果のリスト
        results = [sample_process_result]
        
        # 結果表示関数の呼び出し
        display_results(results)
        
        # DataFrameが呼ばれたことを確認
        assert mock_dataframe.call_count > 0
        
        # エクスパンダーが呼ばれたことを確認
        assert mock_expander.call_count > 0

if __name__ == '__main__':
    unittest.main() 