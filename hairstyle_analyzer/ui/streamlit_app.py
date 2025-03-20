"""
Streamlitアプリケーションモジュール

このモジュールは、ヘアスタイル画像解析システムのStreamlit UIを提供します。
画像アップロード、分析実行、結果表示、エクセル出力などの機能を含みます。
"""

import os
import sys
import logging
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64

# 環境変数の設定
os.environ["PYTHONIOENCODING"] = "utf-8"

# セッションキー
SESSION_SALON_URL = "salon_url"
SESSION_PROCESSOR = "processor"
SESSION_RESULTS = "results"
SESSION_STYLISTS = "stylists"
SESSION_COUPONS = "coupons"
SESSION_PROGRESS = "progress"
SESSION_USE_CACHE = "use_cache"
SESSION_CONFIG = "config"
SESSION_PROCESSING_STAGES = "processing_stages"  # 処理段階を追跡するための新しいセッションキー
SESSION_TEMPLATE_CHOICES = "template_choices"  # テンプレート選択肢を保存するセッションキー
SESSION_USER_SELECTIONS = "user_selections"  # ユーザーの選択を保存するセッションキー

# モジュールのインポート
from hairstyle_analyzer.data.config_manager import ConfigManager
from hairstyle_analyzer.data.template_manager import TemplateManager
from hairstyle_analyzer.data.cache_manager import CacheManager
from hairstyle_analyzer.services.scraper.scraper_service import ScraperService

# コアモジュール
from hairstyle_analyzer.core.template_matcher import TemplateMatcher
from hairstyle_analyzer.core.image_analyzer import ImageAnalyzer
from hairstyle_analyzer.core.style_matching import StyleMatchingService
from hairstyle_analyzer.core.excel_exporter import ExcelExporter
from hairstyle_analyzer.core.processor import MainProcessor
from hairstyle_analyzer.core.text_exporter import TextExporter

# 新しいアーキテクチャに関連するインポート
# ※これらのモジュールパスが現在の構造と一致しない場合は、コメントアウトし、必要に応じて修正します
# from hairstyle_analyzer.services.gemini_service import GeminiService
# from hairstyle_analyzer.analyzer.style_analyzer import StyleAnalyzer
# from hairstyle_analyzer.analyzer.attribute_analyzer import AttributeAnalyzer
# from hairstyle_analyzer.expert.matchmaking_expert import MatchmakingExpert
# from hairstyle_analyzer.recommender.style_recommender import StyleRecommender
# from hairstyle_analyzer.processor.style_processor import StyleProcessor

# 実際に使用可能なインポート（上記の代替として）
from hairstyle_analyzer.services.gemini.gemini_service import GeminiService

# UI コンポーネント
from hairstyle_analyzer.utils.async_context import progress_tracker

from hairstyle_analyzer.data.models import ProcessResult, StyleAnalysis, AttributeAnalysis, Template, StylistInfo, CouponInfo, StyleFeatures

# モジュールレベルでエクスポート
__all__ = [
    'init_session_state',
    'display_results',
    'display_template_choices',
    'display_progress',
    'export_to_excel',
    'export_to_text',
    'download_excel',
    'download_text',
    'render_sidebar',
    'check_app_status'
]

# セッション変数管理のためのヘルパー関数を追加
def get_session_value(key, default_value=None):
    """セッション変数から値を取得するヘルパー関数"""
    return st.session_state.get(key, default_value)

def set_session_value(key, value):
    """セッション変数に値を設定するヘルパー関数"""
    st.session_state[key] = value

def has_session_key(key):
    """セッション変数にキーが存在するか確認するヘルパー関数"""
    return key in st.session_state

def init_session_state():
    """セッションステートを初期化"""
    # セッション変数の初期化
    if not has_session_key(SESSION_PROCESSOR):
        set_session_value(SESSION_PROCESSOR, None)
    if not has_session_key(SESSION_RESULTS):
        set_session_value(SESSION_RESULTS, [])
    if not has_session_key(SESSION_PROGRESS):
        set_session_value(SESSION_PROGRESS, {
            "current": 0,
            "total": 0,
            "message": "",
            "start_time": None,
            "complete": False
        })
    if not has_session_key(SESSION_STYLISTS):
        set_session_value(SESSION_STYLISTS, [])
    if not has_session_key(SESSION_COUPONS):
        set_session_value(SESSION_COUPONS, [])
    if not has_session_key(SESSION_USE_CACHE):
        set_session_value(SESSION_USE_CACHE, False)
    # テンプレート選択肢の初期化
    if not has_session_key(SESSION_TEMPLATE_CHOICES):
        set_session_value(SESSION_TEMPLATE_CHOICES, {})
    # ユーザー選択の初期化
    if not has_session_key(SESSION_USER_SELECTIONS):
        set_session_value(SESSION_USER_SELECTIONS, {})
    # APIキーのセッション変数初期化は削除
    if not has_session_key(SESSION_SALON_URL):
        set_session_value(SESSION_SALON_URL, "")


def update_progress(current, total, message="", stage_details=None):
    """進捗状況の更新"""
    if has_session_key(SESSION_PROGRESS):
        progress = get_session_value(SESSION_PROGRESS)
        progress["current"] = current
        progress["total"] = total
        progress["message"] = message
        
        # 処理段階の詳細情報を追加
        if stage_details:
            progress["stage_details"] = stage_details
        
        # 完了時の処理
        if current >= total and total > 0:
            progress["complete"] = True
        
        set_session_value(SESSION_PROGRESS, progress)


async def process_images(processor, image_paths, stylists=None, coupons=None, use_cache=False):
    """画像を処理して結果を取得する非同期関数"""
    results = []
    total = len(image_paths)
    
    # プロセッサーがNoneの場合、再初期化を試みる
    if processor is None:
        logging.error("プロセッサーがNoneのため、再初期化を試みます")
        try:
            config_manager = get_config_manager()
            processor = create_processor(config_manager)
            if processor is None:
                logging.error("プロセッサーの再初期化に失敗しました")
                return []
            # 再初期化が成功した場合、セッションに保存
            st.session_state[SESSION_PROCESSOR] = processor
            logging.info("プロセッサーの再初期化に成功し、セッションに保存しました")
        except Exception as e:
            logging.error(f"プロセッサーの再初期化中にエラーが発生: {str(e)}")
            return []
    
    # 画像が存在するか確認
    if not image_paths:
        logging.error("画像パスが空です")
        return []
    
    # 処理段階の定義
    processing_stages = [
        "画像読み込み",
        "スタイル分析",
        "テンプレートマッチング",
        "スタイリスト選択",
        "タイトル生成"
    ]
    
    # 進捗状況の初期化
    progress = {
        "current": 0,
        "total": total,
        "message": "初期化中...",
        "start_time": time.time(),
        "complete": False,
        "stage_details": f"準備中: {processing_stages[0]}"
    }
    st.session_state[SESSION_PROGRESS] = progress
    
    try:
        # キャッシュ設定を適用
        processor.use_cache = use_cache
        
        # 各画像を処理
        for i, image_path in enumerate(image_paths):
            try:
                # 進捗状況の更新
                progress["current"] = i
                progress["message"] = f"画像 {i+1}/{total} を処理中..."
                
                # 文字列パスをPathオブジェクトに変換
                path_obj = Path(image_path) if isinstance(image_path, str) else image_path
                
                # ログに記録
                image_name = path_obj.name
                logging.info(f"画像 {image_name} の処理を開始します")
                
                # 処理段階の詳細情報を更新
                stage_details = f"画像: {image_name}\n"
                stage_details += f"現在の段階: {processing_stages[0]}\n"
                stage_details += f"次の段階: {processing_stages[1]}"
                progress["stage_details"] = stage_details
                
                # セッションステートを更新して進捗表示を更新
                st.session_state[SESSION_PROGRESS] = progress
                
                # 明示的な遅延を入れて、UIの更新を確実にする
                await asyncio.sleep(0.1)
                
                # 画像処理
                if stylists and coupons:
                    # スタイリストとクーポンのデータを渡して処理
                    result = await processor.process_single_image(path_obj, stylists, coupons, use_cache=use_cache)
                else:
                    # 基本処理
                    result = await processor.process_single_image(path_obj, use_cache=use_cache)
                
                # 処理段階の詳細情報を更新（完了）
                stage_details = f"画像: {image_name}\n"
                stage_details += f"完了した段階: {', '.join(processing_stages)}\n"
                stage_details += "処理完了"
                progress["stage_details"] = stage_details
                st.session_state[SESSION_PROGRESS] = progress
                
                # 明示的な遅延を入れて、UIの更新を確実にする
                await asyncio.sleep(0.1)
                
                # 結果にファイル名を追加
                if result:
                    if isinstance(result, dict) and 'image_name' not in result:
                        result['image_name'] = image_name
                        result['image_path'] = str(path_obj)
                    results.append(result)
                
            except Exception as e:
                # 個別の画像処理中のエラーをログに記録（処理は続行）
                logging.error(f"画像処理エラー ({image_name}): {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                
                # エラー情報を進捗詳細に追加
                stage_details = f"画像: {image_name}\n"
                stage_details += f"エラー発生: {str(e)}\n"
                stage_details += "次の画像に進みます"
                progress["stage_details"] = stage_details
                st.session_state[SESSION_PROGRESS] = progress
                
                # 明示的な遅延を入れて、UIの更新を確実にする
                await asyncio.sleep(0.1)
                
                continue
        
        # 進捗状況の更新
        progress["current"] = total
        progress["message"] = "処理完了"
        progress["complete"] = True
        progress["stage_details"] = f"全ての画像処理が完了しました。合計: {total}画像"
        st.session_state[SESSION_PROGRESS] = progress
        
        return results
    
    except Exception as e:
        # 全体の処理中のエラーをログに記録
        logging.error(f"画像処理全体でエラーが発生: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # エラー情報を進捗詳細に追加
        if has_session_key(SESSION_PROGRESS):
            progress = get_session_value(SESSION_PROGRESS)
            progress["message"] = f"エラーが発生しました: {str(e)}"
            progress["stage_details"] = f"処理中にエラーが発生しました:\n{str(e)}"
            set_session_value(SESSION_PROGRESS, progress)
        
        # UIの更新を確実にするための遅延
        await asyncio.sleep(0.1)
        
        return []


def create_processor(config_manager):
    """プロセッサーを作成する関数"""
    try:
        logging.info("プロセッサーの作成を開始します")
        
        # 設定マネージャーがNoneの場合の対応
        if config_manager is None:
            logging.error("設定マネージャーがNoneです")
            return None
        
        # テンプレートマネージャーの初期化
        template_manager = TemplateManager(config_manager.paths.template_csv)
        logging.info(f"テンプレートファイル: {config_manager.paths.template_csv}")
        
        # テンプレートマネージャーの初期化確認
        if not template_manager:
            logging.error("テンプレートマネージャーの初期化に失敗しました")
            return None
        
        # キャッシュマネージャーの初期化
        cache_manager = CacheManager(config_manager.paths.cache_file, config_manager.cache)
        logging.info(f"キャッシュファイル: {config_manager.paths.cache_file}")
        
        # APIキーの確認と取得
        api_key = get_api_key()
        if not api_key:
            logging.warning("APIキーが設定されていません。画像処理は機能しません。")
        
        # GeminiServiceの初期化（APIキーを直接コンストラクタに渡す）
        # コンフィグにAPIキーを設定
        config_manager.gemini.api_key = api_key
        
        # APIキーを含むコンフィグでGeminiServiceを初期化
        gemini_service = GeminiService(config_manager.gemini)
        logging.info(f"Gemini API設定: モデル={config_manager.gemini.model}")
        
        # 各コアコンポーネントの初期化
        image_analyzer = ImageAnalyzer(gemini_service, cache_manager)
        template_matcher = TemplateMatcher(template_manager)
        style_matcher = StyleMatchingService(gemini_service)
        excel_exporter = ExcelExporter(config_manager.excel)
        text_exporter = TextExporter(config_manager.text)
        
        # キャッシュ使用設定の取得
        use_cache = st.session_state.get(SESSION_USE_CACHE, True)
        logging.info(f"キャッシュ使用設定: {use_cache}")
        
        # メインプロセッサーの初期化
        processor = MainProcessor(
            image_analyzer=image_analyzer,
            template_matcher=template_matcher,
            style_matcher=style_matcher,
            excel_exporter=excel_exporter,
            text_exporter=text_exporter,
            cache_manager=cache_manager,
            batch_size=config_manager.processing.batch_size,
            api_delay=config_manager.processing.api_delay,
            use_cache=use_cache
        )
        
        logging.info("プロセッサーの作成が完了しました")
        return processor
        
    except Exception as e:
        # エラーの詳細をログに記録
        logging.error(f"プロセッサー作成中にエラーが発生: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def display_progress():
    """進捗状況の表示"""
    if has_session_key(SESSION_PROGRESS):
        progress = get_session_value(SESSION_PROGRESS)
        current = progress["current"]
        total = progress["total"]
        message = progress["message"]
        
        if total > 0:
            # プログレスバーのスタイル改善
            st.markdown("""
            <style>
                .stProgress > div > div {
                    background-color: #4CAF50;
                    transition: width 0.3s ease;
                }
                .progress-label {
                    font-size: 16px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                .progress-details {
                    display: flex;
                    justify-content: space-between;
                    margin-top: 5px;
                    color: #555;
                }
                .stage-indicator {
                    padding: 5px 10px;
                    border-radius: 4px;
                    background-color: #f0f0f0;
                    margin-right: 5px;
                    font-size: 14px;
                }
                .stage-active {
                    background-color: #e6f7ff;
                    border-left: 3px solid #1890ff;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # プログレスバーのラベル表示
            st.markdown('<p class="progress-label">画像処理の進捗状況</p>', unsafe_allow_html=True)
            
            # プログレスバーの表示
            progress_val = min(current / total, 1.0)
            progress_bar = st.progress(progress_val)
            
            # 進捗情報を2カラムで表示
            col1, col2 = st.columns(2)
            
            with col1:
                # 進捗メッセージの表示
                if message:
                    st.write(f"**状態**: {message}")
                
                # 処理数と割合の表示
                percentage = int(progress_val * 100)
                st.write(f"**進捗**: {current}/{total} 画像 ({percentage}%)")
            
            with col2:
                # 処理時間の表示
                if progress["start_time"]:
                    elapsed = time.time() - progress["start_time"]
                    
                    # 経過時間のフォーマット
                    if elapsed < 60:
                        elapsed_str = f"{elapsed:.1f}秒"
                    else:
                        minutes = int(elapsed // 60)
                        seconds = int(elapsed % 60)
                        elapsed_str = f"{minutes}分{seconds}秒"
                    
                    st.write(f"**経過時間**: {elapsed_str}")
                    
                    # 処理速度の計算と表示
                    if current > 0:
                        speed = current / elapsed
                        if speed < 1:
                            st.write(f"**処理速度**: {speed:.2f} 画像/秒")
                        else:
                            st.write(f"**処理速度**: {speed*60:.1f} 画像/分")
                    
                    # 残り時間の予測（現在の進捗から）
                    if 0 < current < total:
                        remaining = (elapsed / current) * (total - current)
                        
                        # 残り時間のフォーマット
                        if remaining < 60:
                            remaining_str = f"{remaining:.1f}秒"
                        else:
                            minutes = int(remaining // 60)
                            seconds = int(remaining % 60)
                            remaining_str = f"{minutes}分{seconds}秒"
                        
                        st.write(f"**推定残り時間**: {remaining_str}")
            
            # 処理段階の表示（折りたたみ可能）
            if "stage_details" in progress:
                with st.expander("処理の詳細を表示", expanded=False):
                    st.write("**現在の処理段階**:")
                    st.write(progress["stage_details"])
            
            # 完了メッセージ
            if progress["complete"]:
                st.success(f"🎉 処理が完了しました: {current}/{total}画像")


def display_template_choices(results):
    """テンプレート選択画面を表示する"""
    if not results:
        st.warning("処理された画像データがありません。")
        return
    
    # セッション状態の初期化 (必要な場合)
    if not has_session_key(SESSION_TEMPLATE_CHOICES):
        set_session_value(SESSION_TEMPLATE_CHOICES, {})
    
    if not has_session_key(SESSION_USER_SELECTIONS):
        set_session_value(SESSION_USER_SELECTIONS, {})
    
    # 画像ごとのテンプレート選択肢を表示
    st.header("テンプレートの選択")
    st.markdown("各画像について、最適なテンプレートを選択してください。AIが選んだ最善のオプションが最初に表示されます。")
    
    # 画像ごとに選択肢を表示
    for i, result in enumerate(results):
        image_name = result.image_name
        
        # すでに選択肢がセッションに保存されていなければ追加
        template_choices = get_session_value(SESSION_TEMPLATE_CHOICES, {})
        
        if image_name not in template_choices:
            # 選択肢を作成: デフォルトと代替テンプレート
            templates = [result.selected_template] + result.alternative_templates
            # 辞書全体を更新せず、特定のキーだけを更新
            template_choices[image_name] = templates
            set_session_value(SESSION_TEMPLATE_CHOICES, template_choices)
        
        # 選択肢を取得
        templates = get_session_value(SESSION_TEMPLATE_CHOICES)[image_name]
        
        # 画像ごとにエクスパンダーを表示
        with st.expander(f"画像 {i+1}: {image_name}", expanded=(i==0)):
            cols = st.columns([1, 2])
            
            with cols[0]:
                # 画像を表示（可能な場合）
                if hasattr(result, 'image_data') and result.image_data:
                    try:
                        st.image(result.image_data, caption=f"画像: {image_name}", use_column_width=True)
                    except Exception as e:
                        st.error(f"画像の表示中にエラーが発生しました: {str(e)}")
                elif hasattr(result, 'image_path') and result.image_path:
                    try:
                        st.image(result.image_path, caption=f"画像: {image_name}", use_column_width=True)
                    except Exception as e:
                        st.error(f"画像ファイルの表示中にエラーが発生しました: {str(e)}")
                else:
                    st.info("画像データが利用できません")
            
            with cols[1]:
                # 選択肢のタイトルをラジオボタン用に準備
                template_titles = []
                for j, template in enumerate(templates):
                    prefix = "🌟 AIおすすめ: " if j == 0 else f"選択肢 {j}: "
                    template_titles.append(f"{prefix}{template.title}")
                
                # ユーザー選択のデフォルト値を設定
                user_selections = get_session_value(SESSION_USER_SELECTIONS, {})
                default_idx = user_selections.get(image_name, 0)
                
                # テンプレート選択用のラジオボタン
                selected_idx = st.radio(
                    "テンプレートを選択してください",
                    options=range(len(template_titles)),
                    format_func=lambda i: template_titles[i],
                    index=default_idx,
                    key=f"template_radio_{image_name}"
                )
                
                # 選択をセッションに保存
                user_selections[image_name] = selected_idx
                set_session_value(SESSION_USER_SELECTIONS, user_selections)
                
                # 選択されたテンプレート情報を表示
                selected_template = templates[selected_idx]
                st.markdown("### 選択されたテンプレート情報")
                st.markdown(f"**カテゴリ:** {selected_template.category}")
                st.markdown(f"**タイトル:** {selected_template.title}")
                st.markdown(f"**メニュー:** {selected_template.menu}")
                st.markdown(f"**コメント:**\n{selected_template.comment}")
                if hasattr(selected_template, 'hashtag'):
                    st.markdown(f"**ハッシュタグ:** {selected_template.hashtag}")
    
    # 確定ボタン
    st.write("すべての選択内容に問題がなければ、確定ボタンを押してください。")
    if st.button("テンプレート選択を確定する", type="primary"):
        # ユーザー選択をもとに、各ResultのテンプレートをUpdated
        for result in results:
            image_name = result.image_name
            if image_name in get_session_value(SESSION_USER_SELECTIONS):
                selected_idx = get_session_value(SESSION_USER_SELECTIONS)[image_name]
                templates = get_session_value(SESSION_TEMPLATE_CHOICES)[image_name]
                if 0 <= selected_idx < len(templates):
                    result.user_selected_template = templates[selected_idx]
        
        # 確定フラグをセット
        set_session_value('confirm_template_selections', True)
        
        # 結果を更新
        set_session_value(SESSION_RESULTS, results)
        
        st.success("テンプレート選択を確定しました!")
        st.balloons()  # 視覚的な演出

def convert_results_to_dataframe(results):
    """処理結果をDataFrameに変換する"""
    try:
        data = []
        for result in results:
            # 画像ごとの情報を抽出
            template = result.selected_template
            row = {
                "画像": result.image_name,
                "カテゴリ": template.category if template else "未選択",
                "スタイル": template.title if template else "未選択",
                "メニュー": template.menu if template else "未設定",
                "コメント": template.comment if template else "未設定",
                "ハッシュタグ": template.hashtag if (template and hasattr(template, 'hashtag')) else "なし"
            }
            data.append(row)
        
        # DataFrameに変換
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"DataFrameへの変換中にエラーが発生しました: {e}")
        st.error(f"データの変換中にエラーが発生しました: {str(e)}")
        return pd.DataFrame()

def display_summary_table(df):
    """結果のサマリーテーブルを表示する"""
    if not df.empty:
        st.subheader("結果サマリー")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("表示するデータがありません")

def display_image_details(results):
    """各画像の詳細情報を表示する"""
    st.subheader("画像ごとの詳細")
    
    for i, result in enumerate(results):
        with st.expander(f"画像 {i+1}: {result.image_name}", expanded=i == 0):
            # 1行2列のレイアウト
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # 画像を表示
                if hasattr(result, 'image_data') and result.image_data is not None:
                    try:
                        # PILイメージをバイトストリームに変換
                        img_byte_arr = io.BytesIO()
                        result.image_data.save(img_byte_arr, format=result.image_data.format or 'JPEG')
                        img_byte_arr.seek(0)
                        
                        # 画像を表示
                        st.image(img_byte_arr, caption=result.image_name, use_column_width=True)
                    except Exception as e:
                        st.error(f"画像の表示中にエラーが発生しました: {str(e)}")
                else:
                    st.warning("画像データがありません")
            
            with col2:
                # 選択されたテンプレート情報を表示
                if result.user_selected_template:
                    template = result.user_selected_template
                    st.write("### 選択されたテンプレート")
                    st.write(f"**カテゴリ:** {template.category}")
                    st.write(f"**タイトル:** {template.title}")
                    st.write(f"**メニュー:** {template.menu}")
                    st.write(f"**コメント:** {template.comment}")
                    # hashtag属性を使用する（模範データのhashtags属性参照をhashtag属性に修正）
                    if hasattr(template, 'hashtag'):
                        st.write(f"**ハッシュタグ:** {template.hashtag}")
                elif result.selected_template:
                    template = result.selected_template
                    st.write("### AIが選択したテンプレート")
                    st.write(f"**カテゴリ:** {template.category}")
                    st.write(f"**タイトル:** {template.title}")
                    st.write(f"**メニュー:** {template.menu}")
                    st.write(f"**コメント:** {template.comment}")
                    # hashtag属性を使用する
                    if hasattr(template, 'hashtag'):
                        st.write(f"**ハッシュタグ:** {template.hashtag}")
                else:
                    st.warning("テンプレートが選択されていません")

def display_export_buttons():
    """エクスポートボタンを表示する"""
    st.subheader("結果のエクスポート")
    
    # 結果とプロセッサをセッションから取得
    results = get_session_value(SESSION_RESULTS, [])
    processor = get_session_value(SESSION_PROCESSOR, None)
    
    if not results or not processor:
        st.warning("エクスポート可能なデータがありません。")
        return
    
    # 2列のレイアウト
    col1, col2 = st.columns(2)
    
    with col1:
        # Excelエクスポートボタン
        generate_excel_download(processor, results, "エクスポート可能")
    
    with col2:
        # テキストエクスポートボタン
        generate_text_download(processor, results, "エクスポート可能")

def display_results(results):
    """処理結果を表示する（分割された関数を呼び出す）"""
    if not results or len(results) == 0:
        st.warning("表示する結果がありません。")
        return
    
    # 結果をDataFrameに変換
    df = convert_results_to_dataframe(results)
    
    # サマリーテーブルの表示
    display_summary_table(df)
    
    # 画像ごとの詳細表示
    display_image_details(results)
    
    # エクスポートボタンの表示
    display_export_buttons()


async def fetch_salon_data(url, config_manager):
    """サロンデータの取得"""
    if not url:
        st.warning("サロンURLを入力してください")
        return None, None
    
    # キャッシュディレクトリの設定
    cache_dir = Path(os.environ.get("CACHE_DIR", "cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "scraper_cache.json"
    
    try:
        # スクレイパーサービスの初期化
        async with ScraperService(
            config=config_manager.scraper,
            cache_path=cache_path
        ) as scraper:
            st.write("サロンデータを取得中...")
            progress_bar = st.progress(0.0)
            
            # スタイリストとクーポン情報の取得
            stylists, coupons = await scraper.fetch_all_data(url)
            
            # 結果保存
            set_session_value(SESSION_STYLISTS, stylists)
            set_session_value(SESSION_COUPONS, coupons)
            
            progress_bar.progress(1.0)
            st.success(f"スタイリスト{len(stylists)}名、クーポン{len(coupons)}件のデータを取得しました。")
            
            return stylists, coupons
        
    except Exception as e:
        st.error(f"サロンデータの取得中にエラーが発生しました: {str(e)}")
        return None, None


def render_sidebar(config_manager):
    """サイドバーの表示"""
    with st.sidebar:
        st.title("設定")
        
        # サロン設定
        st.header("サロン設定")
        salon_url = st.text_input(
            "ホットペッパービューティURL",
            value=get_session_value(SESSION_SALON_URL, config_manager.scraper.base_url),
            help="サロンのホットペッパービューティURLを入力してください。"
        )
        
        # URLをセッションに保存
        if salon_url:
            set_session_value(SESSION_SALON_URL, salon_url)
        
        # サロンデータ取得ボタン
        if st.button("サロンデータを取得"):
            # URLの検証
            if not salon_url or not salon_url.startswith("https://beauty.hotpepper.jp/"):
                st.error("有効なホットペッパービューティURLを入力してください。")
            else:
                # 非同期でサロンデータを取得
                asyncio.run(fetch_salon_data(salon_url, config_manager))
        
        # スタイリストとクーポン情報を表示
        if has_session_key(SESSION_STYLISTS) and has_session_key(SESSION_COUPONS):
            stylists = get_session_value(SESSION_STYLISTS)
            coupons = get_session_value(SESSION_COUPONS)
            
            if stylists:
                st.write(f"スタイリスト: {len(stylists)}人")
                stylist_expander = st.expander("スタイリスト一覧")
                with stylist_expander:
                    for i, stylist in enumerate(stylists[:10]):  # 表示数を制限
                        st.write(f"{i+1}. {stylist.name}")
                    if len(stylists) > 10:
                        st.write(f"...他 {len(stylists) - 10}人")
            
            if coupons:
                st.write(f"クーポン: {len(coupons)}件")
                coupon_expander = st.expander("クーポン一覧")
                with coupon_expander:
                    for i, coupon in enumerate(coupons[:10]):  # 表示数を制限
                        st.write(f"{i+1}. {coupon.name}")
                    if len(coupons) > 10:
                        st.write(f"...他 {len(coupons) - 10}件")
        
        # 詳細設定セクション
        st.header("詳細設定")
        with st.expander("詳細設定"):
            # バッチサイズ設定
            batch_size = st.slider(
                "バッチサイズ",
                min_value=1,
                max_value=10,
                value=config_manager.processing.batch_size,
                help="一度に処理する画像の数です。大きすぎるとメモリ不足になる可能性があります。"
            )
            
            # API遅延設定
            api_delay = st.slider(
                "API遅延（秒）",
                min_value=0.1,
                max_value=5.0,
                value=config_manager.processing.api_delay,
                step=0.1,
                help="API呼び出し間の遅延時間です。小さすぎるとレート制限に達する可能性があります。"
            )
            
            # キャッシュTTL設定
            cache_ttl_days = st.slider(
                "キャッシュ有効期間（日）",
                min_value=1,
                max_value=30,
                value=config_manager.cache.ttl_days,
                help="キャッシュの有効期間です。長すぎると古い結果が返される可能性があります。"
            )
            
            # 設定を保存
            if st.button("設定を保存"):
                try:
                    # 設定の更新
                    config_updates = {
                        "processing": {
                            "batch_size": batch_size,
                            "api_delay": api_delay
                        },
                        "cache": {
                            "ttl_days": cache_ttl_days
                        }
                    }
                    
                    # スクレイパーURLの更新
                    if salon_url:
                        config_updates["scraper"] = {
                            "base_url": salon_url
                        }
                    
                    # 設定の更新
                    config_manager.update_config(config_updates)
                    
                    st.success("設定を保存しました。")
                
                except Exception as e:
                    st.error(f"設定の保存中にエラーが発生しました: {str(e)}")
        
        # キャッシュ管理セクション
        st.header("キャッシュ管理")
        
        # キャッシュ使用設定
        use_cache = st.checkbox(
            "キャッシュを使用する",
            value=get_session_value(SESSION_USE_CACHE, True),
            help="オフにすると毎回APIリクエストを実行します。テスト時などに有用です。"
        )
        
        # キャッシュ使用設定をセッションに保存
        set_session_value(SESSION_USE_CACHE, use_cache)
        
        # プロセッサーがすでに存在する場合は設定を更新
        if has_session_key(SESSION_PROCESSOR) and get_session_value(SESSION_PROCESSOR) is not None:
            processor = get_session_value(SESSION_PROCESSOR)
            processor.set_use_cache(use_cache)
            set_session_value(SESSION_PROCESSOR, processor)
        

def convert_to_process_results(results):
    """結果をProcessResultオブジェクトに変換する関数"""
    from hairstyle_analyzer.data.models import ProcessResult, StyleAnalysis, AttributeAnalysis, Template, StylistInfo, CouponInfo, StyleFeatures
    
    process_results = []
    for result in results:
        try:
            if isinstance(result, dict):
                # 辞書の場合はProcessResultオブジェクトに変換
                # 必要なオブジェクトを作成
                image_name = result.get("image_name", "")
                
                # StyleAnalysisの作成
                style_analysis_dict = result.get("style_analysis", {})
                features_dict = style_analysis_dict.get("features", {}) if isinstance(style_analysis_dict, dict) else {}
                
                features = StyleFeatures(
                    color=features_dict.get("color", ""),
                    cut_technique=features_dict.get("cut_technique", ""),
                    styling=features_dict.get("styling", ""),
                    impression=features_dict.get("impression", "")
                )
                
                style_analysis = StyleAnalysis(
                    category=style_analysis_dict.get("category", "") if isinstance(style_analysis_dict, dict) else "",
                    features=features,
                    keywords=style_analysis_dict.get("keywords", []) if isinstance(style_analysis_dict, dict) else []
                )
                
                # AttributeAnalysisの作成
                attribute_analysis_dict = result.get("attribute_analysis", {})
                attribute_analysis = AttributeAnalysis(
                    sex=attribute_analysis_dict.get("sex", "") if isinstance(attribute_analysis_dict, dict) else "",
                    length=attribute_analysis_dict.get("length", "") if isinstance(attribute_analysis_dict, dict) else ""
                )
                
                # Templateの作成
                template_dict = result.get("selected_template", {})
                template = Template(
                    category=template_dict.get("category", "") if isinstance(template_dict, dict) else "",
                    title=template_dict.get("title", "") if isinstance(template_dict, dict) else "",
                    menu=template_dict.get("menu", "") if isinstance(template_dict, dict) else "",
                    comment=template_dict.get("comment", "") if isinstance(template_dict, dict) else "",
                    hashtag=template_dict.get("hashtag", "") if isinstance(template_dict, dict) else ""
                )
                
                # StylistInfoの作成
                stylist_dict = result.get("selected_stylist", {})
                stylist = StylistInfo(
                    name=stylist_dict.get("name", "") if isinstance(stylist_dict, dict) else "",
                    specialties=stylist_dict.get("specialties", "") if isinstance(stylist_dict, dict) else "",
                    description=stylist_dict.get("description", "") if isinstance(stylist_dict, dict) else ""
                )
                
                # CouponInfoの作成
                coupon_dict = result.get("selected_coupon", {})
                coupon = CouponInfo(
                    name=coupon_dict.get("name", "") if isinstance(coupon_dict, dict) else "",
                    price=coupon_dict.get("price", 0) if isinstance(coupon_dict, dict) else 0,
                    description=coupon_dict.get("description", "") if isinstance(coupon_dict, dict) else ""
                )
                
                # ProcessResultの作成
                process_result = ProcessResult(
                    image_name=image_name,
                    style_analysis=style_analysis,
                    attribute_analysis=attribute_analysis,
                    selected_template=template,
                    selected_stylist=stylist,
                    selected_coupon=coupon,
                    stylist_reason=result.get("stylist_reason", ""),
                    coupon_reason=result.get("coupon_reason", ""),
                    template_reason=result.get("template_reason", "")
                )
                
                process_results.append(process_result)
            else:
                # すでにProcessResultオブジェクトの場合はそのまま追加
                process_results.append(result)
        except Exception as e:
            logging.error(f"結果変換中にエラーが発生しました: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            # エラーが発生しても他の結果を続行
            continue
    
    return process_results

def generate_excel_data(results):
    """結果データからExcelバイナリデータを生成する"""
    try:
        # プロセッサーからエクセルエクスポーターを取得
        processor = get_session_value(SESSION_PROCESSOR)
        if not processor or not hasattr(processor, 'excel_exporter'):
            logging.error("エクセルエクスポーターが初期化されていません。")
            return None
        
        # 結果がある場合のみエクスポート
        if not results or len(results) == 0:
            logging.warning("結果が空のため、エクスポートできません。")
            return None
        
        # エクセルデータを生成
        excel_data = processor.excel_exporter.export(results)
        if not excel_data:
            logging.warning("エクセルデータの生成に失敗しました。")
            return None
        
        return excel_data
    except Exception as e:
        logging.exception(f"エクセルエクスポート中にエラーが発生しました: {e}")
        return None

def generate_text_data(results):
    """結果データからテキストデータを生成する"""
    try:
        # プロセッサーからテキストエクスポーターを取得
        processor = get_session_value(SESSION_PROCESSOR)
        if not processor or not hasattr(processor, 'text_exporter'):
            st.error("テキストエクスポーターが初期化されていません。")
            return None
        
        # 結果をProcessResultオブジェクトに変換
        process_results = convert_to_process_results(results)
        
        # テキストデータを生成
        text_data = processor.text_exporter.get_text_content(process_results)
        
        return text_data
    except Exception as e:
        logging.error(f"テキスト変換中にエラーが発生: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        st.error(f"テキスト生成エラー: {str(e)}")
        return None

def download_excel(excel_data):
    """Excelファイルをダウンロードするボタンを表示する関数"""
    if excel_data is None:
        st.warning("ダウンロードするExcelデータがありません。")
        return False
    
    try:
        # ファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hairstyle_analysis_{timestamp}.xlsx"
        
        # ダウンロードボタンを表示
        st.download_button(
            label="⬇️ Excelをダウンロード",
            data=excel_data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="クリックしてExcelファイルをダウンロード",
        )
        
        return True
    except Exception as e:
        logging.error(f"Excelダウンロード中にエラーが発生: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        st.error(f"Excelダウンロードエラー: {str(e)}")
        return False

def download_text(text_data):
    """テキストファイルをダウンロードするボタンを表示する関数"""
    if text_data is None:
        st.warning("ダウンロードするテキストデータがありません。")
        return False
    
    try:
        # ファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hairstyle_analysis_{timestamp}.txt"
        
        # ダウンロードボタンを表示
        st.download_button(
            label="⬇️ テキストをダウンロード",
            data=text_data,
            file_name=filename,
            mime="text/plain",
            help="クリックしてテキストファイルをダウンロード",
        )
        
        return True
    except Exception as e:
        logging.error(f"テキストダウンロード中にエラーが発生: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        st.error(f"テキストダウンロードエラー: {str(e)}")
        return False

def generate_excel_download(processor, results, title="タイトル生成が完了しました。"):
    """プロセッサーを使用してExcelファイルを生成し、ダウンロードボタンを表示する関数"""
    try:
        # Excelデータを生成または取得
        excel_data = None
        
        if processor and hasattr(processor, 'excel_exporter'):
            # 結果をProcessResultオブジェクトに変換してプロセッサーに追加
            process_results = convert_to_process_results(results)
            
            # 直接エクスポーターを使用してExcelデータを生成
            excel_data = processor.excel_exporter.get_binary_data(process_results)
        else:
            # プロセッサーがない場合は警告を表示
            st.warning("Excel出力のためのプロセッサーが利用できません。")
            return False
        
        if not excel_data:
            return False
        
        # Excel用ダウンロードボタンを表示
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hairstyle_analysis_{timestamp}.xlsx"
        
        # 目立つスタイルでダウンロードボタンを表示（カラム数を2にする）
        col1, col2 = st.columns([1, 2])
        with col2:
            st.download_button(
                label="⬇️ Excelファイルをダウンロード ⬇️",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="クリックしてExcelファイルをダウンロード",
                type="primary",
                use_container_width=True
            )
        
        # 少しスペースを追加
        st.write("")
        
        return True
    
    except Exception as e:
        logging.error(f"Excel出力中にエラーが発生しました: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        st.error(f"Excel出力中にエラーが発生しました: {str(e)}")
        return False

def generate_text_download(processor, results, title="タイトル生成が完了しました。"):
    """プロセッサーを使用してテキストファイルを生成し、ダウンロードボタンを表示する関数"""
    try:
        # テキストデータを生成または取得
        text_data = None
        
        if processor and hasattr(processor, 'text_exporter'):
            # 結果をProcessResultオブジェクトに変換してプロセッサーに追加
            process_results = convert_to_process_results(results)
            
            # 直接エクスポーターを使用してテキストデータを生成
            text_data = processor.text_exporter.get_text_content(process_results)
        else:
            # プロセッサーがない場合は警告を表示
            st.warning("テキスト出力のためのプロセッサーが利用できません。")
            return False
        
        if not text_data:
            return False
        
        # テキスト用ダウンロードボタンを表示
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hairstyle_analysis_{timestamp}.txt"
        
        # 目立つスタイルでダウンロードボタンを表示（カラム数を2にする）
        col1, col2 = st.columns([1, 2])
        with col2:
            st.download_button(
                label="⬇️ テキストファイルをダウンロード ⬇️",
                data=text_data,
                file_name=filename,
                mime="text/plain",
                help="クリックしてテキストファイルをダウンロード",
                type="primary",
                use_container_width=True
            )
        
        # 少しスペースを追加
        st.write("")
        
        return True
    
    except Exception as e:
        logging.error(f"テキスト出力中にエラーが発生しました: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        st.error(f"テキスト出力中にエラーが発生しました: {str(e)}")
        return False

def render_main_content():
    """メインコンテンツを表示する関数"""
    
    # 必要な関数をローカルスコープにインポート（名前解決エラー回避のため）
    from hairstyle_analyzer.ui.streamlit_app import convert_to_process_results, generate_excel_download, generate_text_download
    
    # タイトル表示
    st.write("# Style Generator")
    
    # 説明テキスト
    st.markdown("""
    このアプリケーションは、ヘアスタイル画像を分析し、最適なタイトル、説明、スタイリスト、クーポンを提案します。
    サロン情報を取得してから、画像をアップロードして「タイトル生成」ボタンをクリックしてください。
    """)
    
    # 画像アップロード部分
    uploaded_files = st.file_uploader(
        "ヘアスタイル画像をアップロードしてください",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="PNG, JPG, JPEGフォーマットの画像ファイルをアップロードできます。"
    )
    
    # アップロードされた画像のプレビュー表示
    if uploaded_files:
        st.write(f"{len(uploaded_files)}枚の画像がアップロードされました")
        
        # 画像プレビューを表示（横に並べる）- 列数を4に増やし、画像サイズを制限
        cols = st.columns(min(4, len(uploaded_files)))
        for i, uploaded_file in enumerate(uploaded_files[:8]):  # 最大8枚まで表示
            with cols[i % 4]:
                # 画像を開いてリサイズ
                image = Image.open(uploaded_file)
                # 画像の最大幅を200pxに制限
                st.image(image, caption=uploaded_file.name, width=200)
        
        # 8枚以上の場合は省略メッセージを表示
        if len(uploaded_files) > 8:
            st.write(f"他 {len(uploaded_files) - 8} 枚の画像は省略されています")
        
        # 処理開始ボタン
        if st.button("タイトル生成", type="primary"):
            # セッションからプロセッサーを取得または初期化
            try:
                # プロセッサーが存在するか確認
                if not has_session_key(SESSION_PROCESSOR) or get_session_value(SESSION_PROCESSOR) is None:
                    logging.info("プロセッサーがセッションに存在しないため、新規作成します")
                    config_manager = get_config_manager()
                    processor = create_processor(config_manager)
                    
                    # 初期化に成功したか確認
                    if processor is None:
                        st.error("プロセッサーの初期化に失敗しました。ログを確認してください。")
                        return
                    
                    # セッションに保存
                    set_session_value(SESSION_PROCESSOR, processor)
                    logging.info("プロセッサーを初期化してセッションに保存しました")
                else:
                    processor = get_session_value(SESSION_PROCESSOR)
                    logging.info("セッションからプロセッサーを取得しました")
                
                # 一時ディレクトリに画像を保存
                temp_dir = Path(os.environ.get("TEMP_DIR", "temp")) / "hairstyle_analyzer"
                temp_dir.mkdir(parents=True, exist_ok=True)
                image_paths = handle_image_upload(uploaded_files)
                
                if not image_paths:
                    st.error("画像の保存中にエラーが発生しました。")
                    return
                
                logging.info(f"{len(image_paths)}枚の画像を一時ディレクトリに保存しました")
                
                # プログレスバーの表示
                progress_container = st.container()
                with progress_container:
                    # プログレスバーのスタイル改善
                    st.markdown("""
                    <style>
                        .stProgress > div > div {
                            background-color: #4CAF50;
                            transition: width 0.3s ease;
                        }
                        .progress-label {
                            font-size: 16px;
                            font-weight: bold;
                            margin-bottom: 5px;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # プログレスバーのラベル表示
                    st.markdown('<p class="progress-label">画像処理の進捗状況</p>', unsafe_allow_html=True)
                    
                    # プログレスバーと状態テキスト
                    progress_bar = st.progress(0)
                    col1, col2 = st.columns(2)
                    status_text = col1.empty()
                    time_text = col2.empty()
                
                # 初期化
                processor = get_session_value(SESSION_PROCESSOR)
                
                # 非同期処理を実行
                with st.spinner("画像を処理中..."):
                    # 進捗コールバック関数
                    def update_progress_callback(current, total, message=""):
                        # セッションから最新の進捗情報を取得
                        if has_session_key(SESSION_PROGRESS):
                            progress_data = get_session_value(SESSION_PROGRESS)
                            # 処理中の画像のインデックス
                            img_index = progress_data.get("current", 0)
                            # 総画像数
                            total_images = progress_data.get("total", 1)
                            
                            # 各画像の進捗を5ステップに分割
                            # 画像ごとの処理進捗を計算（0-1の範囲）
                            image_progress = float(current) / float(total) if total > 0 else 0
                            
                            # 全体の進捗を計算（0-1の範囲）
                            # 前の画像はすでに完了（各1.0）、現在の画像は部分的に完了（0.0-1.0）
                            overall_progress = (img_index + image_progress) / total_images
                            
                            # プログレスバーの更新
                            progress_bar.progress(overall_progress)
                            
                            # 進捗状況のテキスト表示
                            percentage = int(overall_progress * 100)
                            status_text.markdown(f"**処理中**: 画像 {img_index+1}/{total_images} ({percentage}%)<br>**状態**: {message}", unsafe_allow_html=True)
                            
                            # 経過時間と推定残り時間の表示
                            if "start_time" in progress_data:
                                elapsed = time.time() - progress_data["start_time"]
                                
                                # 経過時間のフォーマット
                                if elapsed < 60:
                                    elapsed_str = f"{elapsed:.1f}秒"
                                else:
                                    minutes = int(elapsed // 60)
                                    seconds = int(elapsed % 60)
                                    elapsed_str = f"{minutes}分{seconds}秒"
                                
                                time_info = f"**経過時間**: {elapsed_str}<br>"
                                
                                # 処理速度と残り時間の計算（現在の画像も考慮）
                                # 完了した画像 + 現在の画像の進捗
                                completed_progress = img_index + image_progress
                                if completed_progress > 0:
                                    # 1画像あたりの平均秒数
                                    avg_seconds_per_image = elapsed / completed_progress
                                    # 残りの画像数
                                    remaining_images = total_images - completed_progress
                                    # 残り時間の予測
                                    remaining = avg_seconds_per_image * remaining_images
                                    
                                    # 処理速度の表示
                                    images_per_minute = 60 / avg_seconds_per_image
                                    if images_per_minute < 1:
                                        speed_str = f"{images_per_minute*60:.1f} 画像/時間"
                                    else:
                                        speed_str = f"{images_per_minute:.1f} 画像/分"
                                    
                                    time_info += f"**処理速度**: {speed_str}<br>"
                                    
                                    # 残り時間の表示
                                    if remaining < 60:
                                        remaining_str = f"{remaining:.1f}秒"
                                    else:
                                        minutes = int(remaining // 60)
                                        seconds = int(remaining % 60)
                                        remaining_str = f"{minutes}分{seconds}秒"
                                    
                                    time_info += f"**推定残り時間**: {remaining_str}"
                                
                                time_text.markdown(time_info, unsafe_allow_html=True)
                    
                    # スタイリストとクーポンのデータを取得
                    stylists = get_session_value(SESSION_STYLISTS, [])
                    coupons = get_session_value(SESSION_COUPONS, [])
                    
                    # スタイリストとクーポンのデータが存在するか確認
                    if not stylists:
                        st.warning("スタイリスト情報が取得されていません。サイドバーの「サロンデータを取得」ボタンを押してデータを取得してください。")
                    if not coupons:
                        st.warning("クーポン情報が取得されていません。サイドバーの「サロンデータを取得」ボタンを押してデータを取得してください。")
                    
                    # キャッシュ使用設定の取得
                    use_cache = get_session_value(SESSION_USE_CACHE, True)
                    
                    # 処理の実行（スタイリストとクーポンのデータとキャッシュ設定を渡す）
                    # 進捗コールバック関数をセット
                    processor.set_progress_callback(lambda current, total, message: update_progress_callback(current, total, message))
                    results = asyncio.run(process_images(processor, image_paths, stylists, coupons, use_cache))
                    
                    # 処理完了
                    progress_bar.progress(1.0)
                    status_text.markdown("**処理完了**！🎉", unsafe_allow_html=True)
                    
                    # 処理詳細の表示
                    if has_session_key(SESSION_PROGRESS) and "stage_details" in get_session_value(SESSION_PROGRESS):
                        with progress_container.expander("処理の詳細を表示", expanded=False):
                            st.write(get_session_value(SESSION_PROGRESS)["stage_details"])
                    
                    # 結果が空でないか確認
                    if not results:
                        st.error("画像処理中にエラーが発生しました。ログを確認してください。")
                        return
                    
                    # 結果をセッションに保存
                    set_session_value(SESSION_RESULTS, results)
                    
                    # 結果表示
                    display_results(results)
                    
                    # ここから出力処理を追加
                    try:
                        # プロセッサーがセッションに存在することを確認
                        processor = get_session_value(SESSION_PROCESSOR)
                        
                        # 出力前にプロセッサーの結果をクリアして、新しい結果をセット
                        processor.clear_results()
                        process_results = convert_to_process_results(results)
                        processor.results.extend(process_results)
                        
                        # 出力形式の選択を削除し、両方の出力を表示
                        st.write("### 出力ファイル")
                        
                        # 通知メッセージを表示
                        st.success("タイトル生成が完了しました。下のボタンをクリックしてファイルをダウンロードしてください。")
                        
                        # Excel出力とダウンロードボタン表示
                        generate_excel_download(processor, results, "タイトル生成が完了しました。")
                        
                        # テキスト出力とダウンロードボタン表示
                        generate_text_download(processor, results, "タイトル生成が完了しました。")
                    
                    except Exception as e:
                        logging.error(f"ファイル出力中にエラーが発生しました: {str(e)}")
                        import traceback
                        logging.error(traceback.format_exc())
                        st.error(f"ファイル出力中にエラーが発生しました: {str(e)}")
            
            except Exception as e:
                st.error(f"処理中にエラーが発生しました: {str(e)}")
                logging.error(f"処理中にエラーが発生しました: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
    
    # 結果が既にセッションにある場合は表示
    elif has_session_key(SESSION_RESULTS) and get_session_value(SESSION_RESULTS):
        results = get_session_value(SESSION_RESULTS)
        display_results(results)
        
        # プロセッサーがセッションに存在するか確認
        if has_session_key(SESSION_PROCESSOR) and get_session_value(SESSION_PROCESSOR) is not None:
            try:
                # セッションからプロセッサーを取得
                processor = get_session_value(SESSION_PROCESSOR)
                
                # 出力前にプロセッサーの結果をクリアして、新しい結果をセット
                processor.clear_results()
                process_results = convert_to_process_results(results)
                processor.results.extend(process_results)
                
                # 出力形式の選択を削除し、両方の出力を表示
                st.write("### 出力ファイル")
                
                # 通知メッセージを表示
                st.success("以前の処理結果からファイルを生成できます。下のボタンをクリックしてダウンロードしてください。")
                
                # Excel出力とダウンロードボタン表示
                generate_excel_download(processor, results, "以前の処理結果からExcelファイルを生成できます。")
                
                # テキスト出力とダウンロードボタン表示
                generate_text_download(processor, results, "以前の処理結果からテキストファイルを生成できます。")
            
            except Exception as e:
                logging.error(f"既存結果からのファイル出力中にエラーが発生しました: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                st.error(f"ファイル出力中にエラーが発生しました: {str(e)}")


def get_config_manager():
    """設定マネージャーのインスタンスを取得する"""
    # セッションから取得を試みる
    if has_session_key(SESSION_CONFIG):
        return get_session_value(SESSION_CONFIG)
    
    # セッションになければ新規作成
    config_manager = ConfigManager("config/config.yaml")
    set_session_value(SESSION_CONFIG, config_manager)
    return config_manager


def handle_image_upload(uploaded_files):
    """アップロードされた画像ファイルを一時ディレクトリに保存する関数"""
    if not uploaded_files:
        return []
    
    try:
        # 一時ディレクトリの作成
        temp_dir = Path(os.environ.get("TEMP_DIR", "temp")) / "hairstyle_analyzer"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 前回の一時ファイルをクリーンアップ
        try:
            for old_file in temp_dir.glob("*"):
                if old_file.is_file():
                    old_file.unlink()
            logging.info("前回の一時ファイルをクリーンアップしました")
        except Exception as e:
            logging.warning(f"一時ファイルのクリーンアップ中にエラー: {str(e)}")
        
        # 画像ファイルの保存
        image_paths = []
        for i, file in enumerate(uploaded_files):
            try:
                # ファイル名の取得（拡張子を含む）
                original_filename = file.name
                file_ext = Path(original_filename).suffix.lower()
                
                # ファイル拡張子の検証
                if file_ext not in ['.jpg', '.jpeg', '.png']:
                    logging.warning(f"サポートされていないファイル形式: {file_ext}")
                    continue
                
                # 安全なファイル名の生成
                safe_filename = f"styleimg_{i+1}{file_ext}"
                temp_path = temp_dir / safe_filename
                
                # ファイルの保存
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # 画像の検証
                try:
                    img = Image.open(temp_path)
                    img.verify()  # 画像が有効か検証
                    img.close()
                    # 再度開いてサイズを確認
                    with Image.open(temp_path) as img:
                        width, height = img.size
                        if width <= 0 or height <= 0:
                            logging.warning(f"無効な画像サイズ: {width}x{height}, ファイル: {safe_filename}")
                            continue
                        logging.info(f"画像サイズ: {width}x{height}, ファイル: {safe_filename}")
                except Exception as e:
                    logging.error(f"画像検証エラー ({safe_filename}): {str(e)}")
                    continue
                
                # 成功した場合、パスをリストに追加（文字列として）
                image_paths.append(str(temp_path))
                logging.info(f"画像を保存しました: {original_filename} -> {safe_filename}")
                
            except Exception as e:
                logging.error(f"画像アップロードエラー: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        return image_paths
        
    except Exception as e:
        logging.error(f"画像のアップロード全体でエラーが発生: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return []


def get_api_key():
    """APIキーを取得する関数"""
    try:
        # 環境変数からの取得を最初に試みる（.envファイルから）
        if "GEMINI_API_KEY" in os.environ:
            api_key = os.environ["GEMINI_API_KEY"]
            logging.info("環境変数からのキー取得: 成功")
            return api_key
            
        # Streamlit Secretsからの取得を試みる（抑制された警告で）
        # シークレットの存在を事前にチェック
        secrets_path = Path(".streamlit/secrets.toml")
        home_secrets_path = Path.home() / ".streamlit/secrets.toml"
        
        if secrets_path.exists() or home_secrets_path.exists():
            # シークレットファイルが存在する場合のみアクセスを試みる
            try:
                if "GEMINI_API_KEY" in st.secrets:
                    api_key = st.secrets["GEMINI_API_KEY"]
                    logging.info("Streamlit Secretsからのキー取得: 成功")
                    return api_key
            except Exception as e:
                # シークレットアクセスのエラーは抑制する（ログのみ）
                logging.debug(f"シークレットアクセス中の例外（無視します）: {str(e)}")
        else:
            # シークレットファイルが存在しない場合はデバッグログのみ
            logging.debug("secrets.tomlファイルが見つかりません。環境変数のみを使用します。")
        
        # APIキーが見つからなかった場合の処理
        logging.warning("APIキーが設定されていません。.envファイルでGEMINI_API_KEYを設定してください。")
        return None
            
    except Exception as e:
        logging.error(f"APIキー取得中にエラーが発生: {str(e)}")
        return None


def run_streamlit_app(config_manager=None, skip_page_config=False):
    """Streamlitアプリケーションのメインエントリーポイント"""
    # 設定マネージャーを初期化（必要な場合）
    if config_manager is None:
        config_manager = ConfigManager("config/config.yaml")
    
    # ページ設定（skip_page_configがFalseの場合のみ実行）
    if not skip_page_config:
        st.set_page_config(
            page_title="Style Generator",
            page_icon="💇",
            layout="wide",
        )
    
    # 設定マネージャーをグローバル変数として保存
    global _config_manager
    _config_manager = config_manager
    
    # セッション状態の初期化
    init_session_state()
    
    # メイン関数の呼び出し
    main()

def main():
    """メインUI関数"""
    # サイドバーの表示
    config_manager = get_config_manager()
    render_sidebar(config_manager)
    
    # アプリ状態のチェック
    if not check_app_status():
        return
    
    # ファイルアップロードエリア
    uploaded_files = st.file_uploader(
        "ヘアスタイル画像をアップロードしてください（複数可）",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    # セッションで保存した結果の読み込み
    results = get_session_value(SESSION_RESULTS, None)
    
    # 処理ステージを管理（新しい処理の場合はリセット）
    if uploaded_files and not get_session_value(SESSION_PROCESSING_STAGES, {}).get('processing_done', False):
        set_session_value(SESSION_PROCESSING_STAGES, {
            'processing_done': False,
            'template_selection_done': False,
            'results_display_done': False
        })
    
    # 処理ステージのチェック
    processing_stages = get_session_value(SESSION_PROCESSING_STAGES, {})
    
    # ファイルがアップロードされていて、まだ処理されていない場合
    if uploaded_files and not processing_stages.get('processing_done', False):
        if st.button("画像を処理", type="primary"):
            # 画像処理の実行
            with st.spinner("画像を処理しています..."):
                # 画像の一時保存
                image_paths = handle_image_upload(uploaded_files)
                
                # プロセッサの取得または作成
                processor = get_session_value(SESSION_PROCESSOR)
                if processor is None:
                    processor = create_processor(get_config_manager())
                    set_session_value(SESSION_PROCESSOR, processor)
                
                # スタイリストとクーポン情報の取得
                stylists = get_session_value(SESSION_STYLISTS, [])
                coupons = get_session_value(SESSION_COUPONS, [])
                
                # キャッシュ使用設定の取得
                use_cache = get_session_value(SESSION_USE_CACHE, True)
                
                # 進捗表示
                display_progress()
                
                # 画像処理の実行
                results = asyncio.run(process_images(processor, image_paths, stylists, coupons, use_cache))
                
                # 結果をセッションに保存
                set_session_value(SESSION_RESULTS, results)
                
                # 処理ステージを更新
                set_session_value(SESSION_PROCESSING_STAGES, {
                    'processing_done': True,
                    'template_selection_done': False,
                    'results_display_done': False
                })
                
                # 処理後に進捗を更新（完了表示）
                if has_session_key(SESSION_PROGRESS):
                    progress = get_session_value(SESSION_PROGRESS)
                    progress["complete"] = True
                    progress["message"] = "処理完了"
                    set_session_value(SESSION_PROGRESS, progress)
                
                # ページのリロード
                st.rerun()
    
    # テンプレート選択ステージ
    elif results and processing_stages.get('processing_done', False) and not processing_stages.get('template_selection_done', False):
        # テンプレート選択UI表示
        display_template_choices(results)
        
        # 選択が確定されたかチェック
        if get_session_value('confirm_template_selections', False):
            # 処理ステージを更新
            set_session_value(SESSION_PROCESSING_STAGES, {
                'processing_done': True,
                'template_selection_done': True,
                'results_display_done': False
            })
            # フラグをリセット
            set_session_value('confirm_template_selections', False)
            # ページのリロード
            st.rerun()
    
    # 結果表示ステージ
    elif results and processing_stages.get('template_selection_done', True):
        # 最終結果表示
        display_results(results)
        
        # 処理ステージを更新
        if not processing_stages.get('results_display_done', False):
            set_session_value(SESSION_PROCESSING_STAGES, {
                'processing_done': True,
                'template_selection_done': True,
                'results_display_done': True
            })
    
    # 初期状態（まだ画像がアップロードされていない）
    else:
        # 初期メッセージ表示
        if not uploaded_files:
            st.info("画像ファイルをアップロードしてください。")
        
        # 前回の結果がある場合は表示
        if results:
            st.subheader("前回の処理結果")
            display_results(results)

def check_app_status():
    """アプリケーションの状態を確認し、必要に応じて警告やエラーメッセージを表示する関数"""
    # APIキーの確認
    api_key = get_api_key()
    if not api_key:
        st.warning("Gemini APIキーが設定されていません。画像処理機能は動作しません。")
        st.info("""
        APIキーを設定するには:
        1. .envファイルを作成し、`GEMINI_API_KEY=your_key_here`を追加する
        2. または、Streamlit Secretsを使用する
        """)
        # APIキーがなくてもUIの表示は許可
        return True
    
    # テンプレートファイルの確認
    config_manager = get_config_manager()
    if config_manager:
        template_path = Path(config_manager.paths.template_csv)
        if not template_path.exists():
            st.warning(f"テンプレートファイルが見つかりません: {template_path}")
            # テンプレートファイルがなくてもUIの表示は許可
    
    # サロンURLの確認
    salon_url = get_session_value(SESSION_SALON_URL, "")
    if not salon_url:
        # 警告を表示するが、処理は続行
        st.info("サイドバーからサロンURLを設定すると、スタイリストとクーポン情報を取得できます。")
    
    # アプリは正常に動作可能
    return True
