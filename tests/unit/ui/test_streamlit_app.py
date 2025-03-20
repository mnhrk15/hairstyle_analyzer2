"""
Streamlitアプリケーションのユニットテスト
"""

import unittest
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import os
import sys
import streamlit as st
import pandas as pd
import io
import hairstyle_analyzer.ui.streamlit_app
from hairstyle_analyzer.ui.streamlit_app import (
    init_session_state, 
    display_template_choices, 
    display_results,
    main,
    SESSION_RESULTS,
    SESSION_TEMPLATE_CHOICES,
    SESSION_USER_SELECTIONS,
    SESSION_PROCESSING_STAGES,
    render_sidebar
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

# Streamlitのモックを定義
class StreamlitMock:
    """Streamlitのモッククラス"""
    def __init__(self):
        self.session_state = {}
        self.ui_elements = {}
        self.containers = {}
        self.columns = []
        self.expansions = []
        self.buttons = {}
        self.progress_bars = []
    
    def set_session_state(self, key, value):
        """セッション状態を設定"""
        self.session_state[key] = value
    
    def get_session_state(self, key, default=None):
        """セッション状態を取得"""
        return self.session_state.get(key, default)


@pytest.fixture
def mock_streamlit():
    """Streamlitのモック"""
    with patch('streamlit.session_state', new_callable=dict) as mock_session_state:
        with patch('streamlit.title') as mock_title:
            with patch('streamlit.subheader') as mock_subheader:
                with patch('streamlit.write') as mock_write:
                    with patch('streamlit.image') as mock_image:
                        with patch('streamlit.expander') as mock_expander:
                            with patch('streamlit.columns') as mock_columns:
                                with patch('streamlit.radio') as mock_radio:
                                    with patch('streamlit.button') as mock_button:
                                        with patch('streamlit.success') as mock_success:
                                            with patch('streamlit.dataframe') as mock_dataframe:
                                                with patch('streamlit.markdown') as mock_markdown:
                                                    mock_st = StreamlitMock()
                                                    yield mock_st


@pytest.fixture
def mock_config_manager():
    """ConfigManagerのモック"""
    mock_config = MagicMock()
    mock_config.paths.image_folder = "assets/samples"
    return mock_config


@pytest.fixture
def sample_process_results():
    """テスト用の処理結果サンプル"""
    results = []
    
    # サンプル画像名
    image_names = ["styleimg (1).png", "styleimg (2).png"]
    
    for img_name in image_names:
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
        
        # メインテンプレートの作成
        selected_template = Template(
            category="ミディアムボブ",
            title="ふんわりナチュラルミディアムボブ",
            menu="カット+カラー",
            comment="柔らかな質感が魅力的なナチュラルスタイル",
            hashtag="#ナチュラル,#ミディアムボブ,#アッシュブラウン"
        )
        
        # 代替テンプレートの作成
        alternative_templates = [
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
        result = ProcessResult(
            image_name=img_name,
            style_analysis=style_analysis,
            attribute_analysis=attribute_analysis,
            selected_template=selected_template,
            alternative_templates=alternative_templates,
            selected_stylist=stylist,
            selected_coupon=coupon
        )
        
        results.append(result)
    
    return results


@pytest.fixture
def mock_streamlit_extras():
    """Streamlitの追加機能のモック"""
    # rerunをモック
    rerun_mock = MagicMock()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(st, "rerun", rerun_mock)
    monkeypatch.setattr(st, "experimental_rerun", rerun_mock)
    yield rerun_mock
    monkeypatch.undo()


@pytest.mark.usefixtures("mock_streamlit")
class TestStreamlitApp:
    """Streamlitアプリのテストクラス"""
    
    def test_init_session_state(self, monkeypatch):
        """セッション状態の初期化テスト"""
        # セッション状態のモック
        session_state = {}
        monkeypatch.setattr(st, "session_state", session_state)
        
        # 初期化関数の実行
        init_session_state()
        
        # セッション変数が正しく初期化されていることを確認
        assert "processor" in st.session_state
        assert "results" in st.session_state
        assert "progress" in st.session_state
        assert "stylists" in st.session_state
        assert "coupons" in st.session_state
        assert "template_choices" in st.session_state
        assert "user_selections" in st.session_state
    
    @patch('hairstyle_analyzer.ui.streamlit_app.get_config_manager')
    def test_display_template_choices(self, mock_get_config, sample_process_results, monkeypatch):
        """テンプレート選択肢表示のテスト"""
        # sample_process_resultsの内容を確認
        actual_image_count = len(sample_process_results)
        
        # セッション状態のモック
        session_state = {
            SESSION_TEMPLATE_CHOICES: {},
            SESSION_USER_SELECTIONS: {}
        }
        monkeypatch.setattr(st, "session_state", session_state)
        
        # ConfigManagerのモック
        mock_config = MagicMock()
        mock_config.paths.image_folder = "assets/samples"
        mock_get_config.return_value = mock_config
        
        # ラジオボタンの戻り値をモック
        radio_mock = MagicMock(return_value=0)
        monkeypatch.setattr(st, "radio", radio_mock)
        
        # ボタンの戻り値をモック（クリックなし）
        button_mock = MagicMock(return_value=False)
        monkeypatch.setattr(st, "button", button_mock)
        
        # 拡張パネルのモック
        expander_mock = MagicMock()
        monkeypatch.setattr(st, "expander", expander_mock)
        
        # テンプレート選択肢表示関数の実行
        display_template_choices(sample_process_results)
        
        # セッション状態にテンプレート選択肢が保存されていることを確認
        assert len(st.session_state[SESSION_TEMPLATE_CHOICES]) == actual_image_count
        
        # 各画像のテンプレート選択肢を確認
        for result in sample_process_results:
            image_key = result.image_name
            assert image_key in st.session_state[SESSION_TEMPLATE_CHOICES]
            templates = st.session_state[SESSION_TEMPLATE_CHOICES][image_key]
            assert len(templates) == 1 + len(result.alternative_templates)  # メイン1つ + 代替テンプレート
    
    @patch('hairstyle_analyzer.ui.streamlit_app.get_config_manager')
    @patch('streamlit.dataframe')
    @patch('streamlit.columns')
    def test_display_results(self, mock_columns, mock_dataframe, mock_get_config, sample_process_results, monkeypatch):
        """結果表示のテスト"""
        # ConfigManagerのモック
        mock_config = MagicMock()
        mock_config.paths.image_folder = "assets/samples"
        mock_get_config.return_value = mock_config
        
        # DataFrameのモック
        monkeypatch.setattr(st, "dataframe", mock_dataframe)
        
        # 拡張パネルのモック
        expander_mock = MagicMock()
        
        class ExpanderContextManager:
            def __enter__(self):
                return MagicMock()
            def __exit__(self, exc_type, exc_val, exc_tb):
                return False
        
        expander_mock.return_value = ExpanderContextManager()
        monkeypatch.setattr(st, "expander", expander_mock)
        
        # カラムのモック
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        # ボタンモック
        button_mock = MagicMock(return_value=False)
        monkeypatch.setattr(st, "button", button_mock)
        
        # テスト用にユーザー選択をセット
        sample_process_results[0].user_selected_template = sample_process_results[0].alternative_templates[0]
        
        # 結果表示関数の実行
        display_results(sample_process_results)
        
        # DataFrameが呼ばれたことを確認
        assert mock_dataframe.call_count > 0
    
    @patch('hairstyle_analyzer.ui.streamlit_app.check_app_status')
    @patch('hairstyle_analyzer.ui.streamlit_app.get_config_manager')
    @patch('hairstyle_analyzer.ui.streamlit_app.st.file_uploader')
    def test_main_initial_state(self, mock_file_uploader, mock_get_config, mock_check_app_status, monkeypatch):
        """メイン関数の初期状態テスト"""
        # セッション状態のモック
        session_state = {}
        monkeypatch.setattr(st, "session_state", session_state)
        
        # 依存関数のモック
        mock_check_app_status.return_value = True
        mock_file_uploader.return_value = None
        mock_get_config.return_value = MagicMock()
        
        # render_sidebarをモック
        render_sidebar_mock = MagicMock()
        monkeypatch.setattr(hairstyle_analyzer.ui.streamlit_app, "render_sidebar", render_sidebar_mock)
        
        # rerunをモック
        rerun_mock = MagicMock()
        monkeypatch.setattr(st, "rerun", rerun_mock)
        
        # メイン関数の実行
        main()
        
        # ファイルアップローダーが呼ばれたことを確認
        mock_file_uploader.assert_called_once()

    @patch('hairstyle_analyzer.ui.streamlit_app.check_app_status')
    @patch('hairstyle_analyzer.ui.streamlit_app.get_config_manager')
    @patch('hairstyle_analyzer.ui.streamlit_app.st.file_uploader')
    @patch('hairstyle_analyzer.ui.streamlit_app.process_images')
    @patch('hairstyle_analyzer.ui.streamlit_app.st.button')
    def test_main_with_uploads(self, mock_button, mock_process_images, mock_file_uploader, 
                                mock_get_config, mock_check_app_status, 
                                sample_process_results, monkeypatch):
        """アップロードがある場合のメイン関数テスト"""
        # セッション状態のモック
        session_state = {
            'processing_triggered': True,
            SESSION_PROCESSING_STAGES: {
                'processing_done': False,
                'template_selection_done': False,
                'results_display_done': False
            }
        }
        monkeypatch.setattr(st, "session_state", session_state)
        
        # 依存関数のモック
        mock_check_app_status.return_value = True
        mock_file_uploader.return_value = ["test1.jpg", "test2.jpg"]
        mock_process_images.return_value = sample_process_results
        mock_button.return_value = True
        
        # スピナーのモック
        spinner_mock = MagicMock()
        
        class SpinnerContextManager:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc_val, exc_tb):
                return False
        
        spinner_mock.return_value = SpinnerContextManager()
        monkeypatch.setattr(st, "spinner", spinner_mock)
        
        # render_sidebarをモック
        render_sidebar_mock = MagicMock()
        monkeypatch.setattr(hairstyle_analyzer.ui.streamlit_app, "render_sidebar", render_sidebar_mock)
        
        # rerunをモック
        rerun_mock = MagicMock()
        monkeypatch.setattr(st, "rerun", rerun_mock)
        
        # メイン関数の実行
        main()
        
        # 処理結果がセッションに保存されていることを確認
        assert SESSION_RESULTS in st.session_state
        assert st.session_state[SESSION_PROCESSING_STAGES]['processing_done'] == True
        
        # rerunが呼ばれたことを確認
        rerun_mock.assert_called_once()

if __name__ == '__main__':
    unittest.main() 