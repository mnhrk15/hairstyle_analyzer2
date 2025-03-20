"""
処理フローモジュール

このモジュールでは、アプリケーションの中心となる処理フローを定義します。
画像処理、テンプレートマッチング、スタイリスト・クーポン選択、Excel出力などの処理を制御します。
"""

import os
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime

import tqdm.asyncio
from tqdm import tqdm

from ..data.models import ProcessResult, StyleAnalysis, AttributeAnalysis
from ..data.interfaces import (
    ProcessResultProtocol, MainProcessorProtocol, CacheManagerProtocol,
    StyleAnalysisProtocol, AttributeAnalysisProtocol, StylistInfoProtocol, CouponInfoProtocol
)
from ..utils.errors import (
    AppError, ProcessingError, ImageError, GeminiAPIError, 
    ScraperError, TemplateError, ValidationError
)
from ..utils.system_utils import calculate_optimal_batch_size
from ..utils.cache_decorators import cacheable
from ..utils.async_context import progress_tracker
from .image_analyzer import ImageAnalyzer
from .template_matcher import TemplateMatcher
from .style_matching import StyleMatchingService
from .excel_exporter import ExcelExporter
from .text_exporter import TextExporter


class MainProcessor(MainProcessorProtocol):
    """
    メイン処理フロークラス
    
    アプリケーションの中心となる処理フローを制御します。
    画像処理、テンプレートマッチング、スタイリスト・クーポン選択、Excel出力などの処理を統合します。
    """
    
    def __init__(
        self, 
        image_analyzer: ImageAnalyzer,
        template_matcher: TemplateMatcher,
        style_matcher: StyleMatchingService,
        excel_exporter: ExcelExporter,
        text_exporter: TextExporter,
        cache_manager: Optional[CacheManagerProtocol] = None,
        batch_size: int = 5,
        api_delay: float = 1.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_cache: bool = False
    ):
        """
        初期化
        
        Args:
            image_analyzer: 画像分析クラス
            template_matcher: テンプレートマッチングクラス
            style_matcher: スタイルマッチングサービス
            excel_exporter: Excel出力クラス
            text_exporter: テキスト出力クラス
            cache_manager: キャッシュマネージャー（オプション）
            batch_size: バッチサイズ
            api_delay: API呼び出し間の遅延（秒）
            max_retries: 最大リトライ回数
            retry_delay: リトライ間隔（秒）
        """
        self.logger = logging.getLogger(__name__)
        self.image_analyzer = image_analyzer
        self.template_matcher = template_matcher
        self.style_matcher = style_matcher
        self.excel_exporter = excel_exporter
        self.text_exporter = text_exporter
        self.cache_manager = cache_manager
        
        # 設定パラメータ
        self.batch_size = batch_size
        self.api_delay = api_delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_cache = use_cache
        
        # 画像アナライザーにキャッシュ設定を反映
        self.image_analyzer.use_cache = use_cache
        
        # 処理結果
        self.results: List[ProcessResult] = []
        
        # 進捗表示用
        self.progress_callback = None
    
    def set_progress_callback(self, callback) -> None:
        """
        進捗コールバックを設定します。
        非推奨: 代わりに非同期コンテキストマネージャーを使用してください。
        
        Args:
            callback: 進捗状況を通知するコールバック関数
        """
        self.logger.warning("set_progress_callbackは非推奨です。代わりに非同期コンテキストマネージャーを使用してください。")
        self.progress_callback = callback
    
    def _update_progress(self, current: int, total: int, message: str = "") -> None:
        """
        進捗状況を更新します。
        非推奨: 代わりに非同期コンテキストマネージャーを使用してください。
        
        Args:
            current: 現在の進捗
            total: 全体の作業量
            message: 進捗メッセージ
        """
        if self.progress_callback:
            self.progress_callback(current, total, message)
    
    @cacheable(lambda self, image_path, *args, **kwargs: f"process_result:{image_path.name}")
    async def process_single_image(self, image_path: Path, stylists=None, coupons=None) -> Optional[ProcessResultProtocol]:
        """
        単一の画像を処理します。
        
        Args:
            image_path: 画像ファイルのパス
            stylists: スタイリスト情報のリスト（オプション）
            coupons: クーポン情報のリスト（オプション）
            
        Returns:
            処理結果、またはエラー時はNone
            
        Raises:
            ProcessingError: 処理中にエラーが発生した場合
            ImageError: 画像が無効な場合
        """
        self.logger.info(f"画像処理開始: {image_path.name}")
        
        try:
            # 0. 画像読み込み段階
            self._update_progress(0, 5, f"画像読み込み中: {image_path.name}")
            
            # 1. スタイル分析と属性分析を並列実行
            self._update_progress(1, 5, f"スタイル分析中: {image_path.name}")
            categories = self.template_matcher.template_manager.get_all_categories()
            style_analysis, attribute_analysis = await self.image_analyzer.analyze_full(image_path, categories)
            
            if not style_analysis or not attribute_analysis:
                self.logger.error(f"画像分析に失敗しました: {image_path.name}")
                return None
            
            # 2. テンプレートマッチング
            self._update_progress(2, 5, f"テンプレートマッチング中: {image_path.name}")
            template, alternative_templates, template_reason = await self._match_template(image_path, style_analysis)
            
            if not template:
                self.logger.error(f"テンプレートマッチングに失敗しました: {image_path.name}")
                return None
            
            # 3-4. スタイリストとクーポン選択
            self._update_progress(3, 5, f"スタイリスト選択中: {image_path.name}")
            stylist_result = await self._select_stylist(image_path, stylists, style_analysis)
            selected_stylist, stylist_reason = stylist_result if stylist_result else (None, None)
            
            self._update_progress(4, 5, f"クーポン選択中: {image_path.name}")
            coupon_result = await self._select_coupon(image_path, coupons, style_analysis)
            selected_coupon, coupon_reason = coupon_result if coupon_result else (None, None)
            
            # 5. 処理結果作成
            self._update_progress(5, 5, f"タイトル生成中: {image_path.name}")
            result = self._create_process_result(
                image_path=image_path,
                style_analysis=style_analysis,
                attribute_analysis=attribute_analysis,
                template=template,
                alternative_templates=alternative_templates,
                template_reason=template_reason,
                stylist=selected_stylist,
                stylist_reason=stylist_reason,
                coupon=selected_coupon,
                coupon_reason=coupon_reason
            )
            
            return result
            
        except (ProcessingError, GeminiAPIError, ImageError) as e:
            self.logger.error(f"画像処理エラー: {str(e)}")
            raise ProcessingError(f"画像処理中にエラーが発生しました: {str(e)}", image_path=str(image_path)) from e
            
        except Exception as e:
            self.logger.error(f"予期しないエラー: {str(e)}")
            return None
    
    async def _match_template(self, image_path: Path, style_analysis: StyleAnalysisProtocol) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
        """
        テンプレートマッチングを実行
        
        Args:
            image_path: 画像ファイルのパス
            style_analysis: スタイル分析結果
            
        Returns:
            (選択されたテンプレート, 代替テンプレートリスト, 選択理由)のタプル
        """
        # Gemini APIサービスを取得
        gemini_service = self.image_analyzer.gemini_service
        
        # 設定からAIマッチングが有効かどうかを確認
        ai_matching_enabled = gemini_service.config.template_matching.enabled
        fallback_on_failure = gemini_service.config.template_matching.fallback_on_failure
        use_category_filter = gemini_service.config.template_matching.use_category_filter
        max_templates = gemini_service.config.template_matching.max_templates
        
        template = None
        alternative_templates = []
        template_reason = None
        ai_matching_success = False
        
        if ai_matching_enabled:
            self.logger.info("AIベースのテンプレートマッチングを実行します")
            template, alternative_templates, template_reason, ai_matching_success = await self.template_matcher.find_best_template_with_ai(
                image_path=image_path,
                gemini_service=gemini_service,
                analysis=style_analysis,
                use_category_filter=use_category_filter,
                max_templates=max_templates
            )
        
        # AIマッチングが失敗または無効の場合、従来のスコアリングベースのマッチングを使用
        if not ai_matching_success and (fallback_on_failure or not ai_matching_enabled):
            self.logger.info("従来のスコアリングベースのテンプレートマッチングを実行します")
            template = self.template_matcher.find_best_template(style_analysis)
            template_reason = "スコアリングベースのマッチングにより選択されました"
            
            # 代替テンプレートも取得
            alternative_templates = self.template_matcher.find_alternative_templates(style_analysis, count=2)
        
        return template, alternative_templates, template_reason
    
    async def _select_stylist(self, image_path: Path, stylists: List[StylistInfoProtocol], style_analysis: StyleAnalysisProtocol) -> Optional[Tuple[StylistInfoProtocol, str]]:
        """スタイリスト選択を実行"""
        if not stylists or len(stylists) == 0:
            return None
            
        selected_stylist, stylist_reason = await self.style_matcher.select_stylist(
            image_path, stylists, style_analysis
        )
        self.logger.info(f"スタイリスト選択: {selected_stylist.name if selected_stylist else 'なし'}")
        return (selected_stylist, stylist_reason) if selected_stylist else None
    
    async def _select_coupon(self, image_path: Path, coupons: List[CouponInfoProtocol], style_analysis: StyleAnalysisProtocol) -> Optional[Tuple[CouponInfoProtocol, str]]:
        """クーポン選択を実行"""
        if not coupons or len(coupons) == 0:
            return None
            
        selected_coupon, coupon_reason = await self.style_matcher.select_coupon(
            image_path, coupons, style_analysis
        )
        self.logger.info(f"クーポン選択: {selected_coupon.name if selected_coupon else 'なし'}")
        return (selected_coupon, coupon_reason) if selected_coupon else None
    
    def _create_process_result(self, 
                              image_path: Path, 
                              style_analysis: StyleAnalysisProtocol,
                              attribute_analysis: AttributeAnalysisProtocol,
                              template: Dict[str, Any],
                              alternative_templates: List[Dict[str, Any]],
                              template_reason: str,
                              stylist: Optional[StylistInfoProtocol] = None,
                              stylist_reason: Optional[str] = None,
                              coupon: Optional[CouponInfoProtocol] = None,
                              coupon_reason: Optional[str] = None) -> ProcessResultProtocol:
        """
        処理結果オブジェクトを作成
        
        Args:
            image_path: 画像ファイルのパス
            style_analysis: スタイル分析結果
            attribute_analysis: 属性分析結果
            template: 選択されたテンプレート
            alternative_templates: 代替テンプレートのリスト
            template_reason: テンプレート選択理由
            stylist: 選択されたスタイリスト（オプション）
            stylist_reason: スタイリスト選択理由（オプション）
            coupon: 選択されたクーポン（オプション）
            coupon_reason: クーポン選択理由（オプション）
            
        Returns:
            処理結果オブジェクト
        """
        from ..data.models import StyleAnalysis, StyleFeatures, AttributeAnalysis, Template, StylistInfo, CouponInfo
        
        # スタイル分析モデルの作成
        style_features = StyleFeatures(
            color=style_analysis.features.color,
            cut_technique=style_analysis.features.cut_technique,
            styling=style_analysis.features.styling,
            impression=style_analysis.features.impression
        )
        
        style_analysis_model = StyleAnalysis(
            category=style_analysis.category,
            features=style_features,
            keywords=style_analysis.keywords
        )
        
        # 属性分析モデルの作成
        attribute_analysis_model = AttributeAnalysis(
            sex=attribute_analysis.sex,
            length=attribute_analysis.length
        )
        
        # テンプレートモデルの作成
        selected_template = Template(
            category=template.category,
            title=template.title,
            menu=template.menu,
            comment=template.comment,
            hashtag=template.hashtag
        )
        
        # 代替テンプレートモデルリストの作成
        alternative_template_models = []
        for alt_template in alternative_templates:
            alt_template_model = Template(
                category=alt_template.category,
                title=alt_template.title,
                menu=alt_template.menu,
                comment=alt_template.comment,
                hashtag=alt_template.hashtag
            )
            alternative_template_models.append(alt_template_model)
        
        # スタイリストモデルの作成（存在する場合）
        selected_stylist = None
        if stylist:
            selected_stylist = StylistInfo(
                name=stylist.name,
                specialties=getattr(stylist, 'specialties', ''),
                description=stylist.description
            )
        
        # クーポンモデルの作成（存在する場合）
        selected_coupon = None
        if coupon:
            selected_coupon = CouponInfo(
                name=coupon.name,
                price=int(coupon.price) if hasattr(coupon, 'price') and coupon.price and str(coupon.price).isdigit() else 0,
                description=getattr(coupon, 'description', ''),
                categories=getattr(coupon, 'categories', []),
                conditions=getattr(coupon, 'conditions', {})
            )
        
        # ProcessResultモデルの作成
        from ..data.models import ProcessResult
        return ProcessResult(
            image_name=image_path.name,
            style_analysis=style_analysis_model,
            attribute_analysis=attribute_analysis_model,
            selected_template=selected_template,
            alternative_templates=alternative_template_models,
            user_selected_template=None,  # ユーザー選択は初期状態ではNone
            selected_stylist=selected_stylist,
            selected_coupon=selected_coupon,
            stylist_reason=stylist_reason,
            coupon_reason=coupon_reason,
            template_reason=template_reason
        )
    
    async def process_images(self, image_paths: List[Path], use_cache: Optional[bool] = None) -> List[ProcessResultProtocol]:
        """
        複数の画像を処理します。
        
        Args:
            image_paths: 画像ファイルのパスリスト
            use_cache: キャッシュを使用するかどうか（Noneの場合はインスタンスの設定を使用）
            
        Returns:
            処理結果のリスト
        """
        self.logger.info(f"複数画像処理開始: {len(image_paths)}枚")
        
        # 結果リストをクリア
        self.results = []
        
        if not image_paths:
            self.logger.warning("処理する画像がありません")
            return []
        
        # 最適なバッチサイズを計算
        optimal_batch_size = calculate_optimal_batch_size(
            memory_per_item_mb=5,  # 1画像あたりの推定メモリ使用量
            max_batch_size=self.batch_size
        )
        
        self.logger.info(f"バッチサイズを設定: {optimal_batch_size}")
        
        # 画像をバッチに分割
        total_images = len(image_paths)
        batch_count = (total_images + optimal_batch_size - 1) // optimal_batch_size
        batches = [image_paths[i:i + optimal_batch_size] for i in range(0, total_images, optimal_batch_size)]
        
        # キャッシュを使用するかどうかの判定
        should_use_cache = self.use_cache if use_cache is None else use_cache
        
        # 非同期コンテキストマネージャーを使用して進捗を追跡
        async def progress_handler(current, total, message):
            # このメソッドは、後方互換性のために進捗コールバックを呼び出します
            self._update_progress(current, total, message)
        
        processed_count = 0
        
        # 進捗トラッカーを使用して処理を実行
        async with progress_tracker(total_images, progress_handler) as tracker:
            # バッチごとに処理
            for batch_index, batch in enumerate(batches):
                batch_message = f"バッチ {batch_index + 1}/{batch_count} 処理中"
                self.logger.info(batch_message)
                tracker.update(processed_count, batch_message)
                
                # バッチ内の画像を並列処理
                tasks = [self.process_single_image(image_path, use_cache=should_use_cache) for image_path in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 結果の処理
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        # エラーをログに記録
                        self.logger.error(f"画像処理中にエラーが発生しました: {str(result)}")
                        # エラーが発生した画像のパスを取得
                        error_image = batch[i].name if i < len(batch) else "不明"
                        tracker.update(processed_count + i + 1, f"エラー: {error_image}")
                    elif result:
                        # 正常な結果を追加
                        self.results.append(result)
                    
                    processed_count += 1
                    tracker.update(processed_count)
        
        # 結果を返す
        return self.results
    
    async def process_images_with_external_data(
        self, 
        image_paths: List[Path],
        stylists: List[StylistInfoProtocol],
        coupons: List[CouponInfoProtocol],
        use_cache: Optional[bool] = None
    ) -> List[ProcessResultProtocol]:
        """
        外部データ（スタイリスト・クーポン情報）を使用して複数の画像を処理します。
        
        Args:
            image_paths: 画像ファイルのパスリスト
            stylists: スタイリスト情報のリスト
            coupons: クーポン情報のリスト
            
        Returns:
            処理結果のリスト
        """
        self.logger.info(f"外部データを使用した複数画像処理開始: {len(image_paths)}枚")
        
        # 結果リストをクリア
        self.results = []
        
        if not image_paths:
            self.logger.warning("処理する画像がありません")
            return []
        
        if not stylists:
            self.logger.warning("スタイリスト情報がありません")
            return []
        
        if not coupons:
            self.logger.warning("クーポン情報がありません")
            return []
        
        # 最適なバッチサイズを計算
        optimal_batch_size = calculate_optimal_batch_size(
            memory_per_item_mb=5,  # 1画像あたりの推定メモリ使用量
            max_batch_size=self.batch_size
        )
        
        self.logger.info(f"バッチサイズを設定: {optimal_batch_size}")
        
        # 画像をバッチに分割
        total_images = len(image_paths)
        batch_count = (total_images + optimal_batch_size - 1) // optimal_batch_size
        batches = [image_paths[i:i + optimal_batch_size] for i in range(0, total_images, optimal_batch_size)]
        
        # キャッシュを使用するかどうかの判定
        should_use_cache = self.use_cache if use_cache is None else use_cache
        
        # 非同期コンテキストマネージャーを使用して進捗を追跡
        async def progress_handler(current, total, message):
            # このメソッドは、後方互換性のために進捗コールバックを呼び出します
            self._update_progress(current, total, message)
        
        processed_count = 0
        
        # 進捗トラッカーを使用して処理を実行
        async with progress_tracker(total_images, progress_handler) as tracker:
            # バッチごとに処理
            for batch_index, batch in enumerate(batches):
                batch_message = f"バッチ {batch_index + 1}/{batch_count} 処理中"
                self.logger.info(batch_message)
                tracker.update(processed_count, batch_message)
                
                # バッチ内の各画像を順次処理
                for image_path in batch:
                    try:
                        # キャッシュを使用するかどうかの判定
                        should_use_cache = self.use_cache if use_cache is None else use_cache
                        
                        # キャッシュチェック（キャッシュを使用する場合のみ）
                        if should_use_cache and self.cache_manager:
                            cache_key = f"process_result_ext:{image_path.name}"
                            cached_result = self.cache_manager.get(cache_key)
                            if cached_result:
                                self.logger.info(f"キャッシュから処理結果を取得: {image_path.name}")
                                self.results.append(cached_result)
                                processed_count += 1
                                tracker.update(processed_count, f"キャッシュ: {image_path.name}")
                                continue
                        
                        # 1. スタイル分析と属性分析を並列実行
                        categories = self.template_matcher.template_manager.get_all_categories()
                        style_analysis, attribute_analysis = await self.image_analyzer.analyze_full(image_path, categories, use_cache=should_use_cache)
                        
                        if not style_analysis or not attribute_analysis:
                            self.logger.error(f"画像分析に失敗しました: {image_path.name}")
                            processed_count += 1
                            tracker.update(processed_count, f"分析失敗: {image_path.name}")
                            continue
                        
                        # 2. テンプレートマッチング
                        template, alternative_templates, template_reason = await self._match_template(image_path, style_analysis)
                        
                        if not template:
                            self.logger.error(f"テンプレートマッチングに失敗しました: {image_path.name}")
                            processed_count += 1
                            tracker.update(processed_count, f"マッチング失敗: {image_path.name}")
                            continue
                        
                        # 3. スタイリスト選択
                        stylist_result = await self.style_matcher.select_stylist(
                            image_path, stylists, style_analysis
                        )
                        selected_stylist, stylist_reason = stylist_result
                        
                        # 4. クーポン選択
                        coupon_result = await self.style_matcher.select_coupon(
                            image_path, coupons, style_analysis
                        )
                        selected_coupon, coupon_reason = coupon_result
                        
                        # 5. 処理結果の作成
                        try:
                            # スタイル分析が辞書型の場合はStyleAnalysisに変換
                            if isinstance(style_analysis, dict):
                                from ..data.models import StyleAnalysis, StyleFeatures
                                features = StyleFeatures(**style_analysis.get('features', {}))
                                style_analysis = StyleAnalysis(
                                    category=style_analysis.get('category', ''),
                                    features=features,
                                    keywords=style_analysis.get('keywords', [])
                                )
                            
                            # 属性分析が辞書型の場合はAttributeAnalysisに変換
                            if isinstance(attribute_analysis, dict):
                                from ..data.models import AttributeAnalysis
                                attribute_analysis = AttributeAnalysis(
                                    sex=attribute_analysis.get('sex', ''),
                                    length=attribute_analysis.get('length', '')
                                )
                            
                            # スタイリストが辞書型の場合はStylistInfoに変換
                            if isinstance(selected_stylist, dict):
                                from ..data.models import StylistInfo
                                selected_stylist = StylistInfo(
                                    name=selected_stylist.get('name', 'サンプルスタイリスト'),
                                    specialties=selected_stylist.get('specialties', ''),
                                    description=selected_stylist.get('description', '')
                                )
                            
                            # クーポンが辞書型の場合はCouponInfoに変換
                            if isinstance(selected_coupon, dict):
                                from ..data.models import CouponInfo
                                selected_coupon = CouponInfo(
                                    name=selected_coupon.get('name', 'サンプルクーポン'),
                                    price=selected_coupon.get('price', 0),
                                    description=selected_coupon.get('description', ''),
                                    categories=selected_coupon.get('categories', []),
                                    conditions=selected_coupon.get('conditions', {})
                                )
                            
                            # テンプレートが辞書型の場合はTemplateに変換
                            if isinstance(template, dict):
                                from ..data.models import Template
                                template = Template(
                                    category=template.get('category', ''),
                                    title=template.get('title', ''),
                                    menu=template.get('menu', ''),
                                    comment=template.get('comment', ''),
                                    hashtag=template.get('hashtag', '')
                                )
                            
                            result = ProcessResult(
                                image_name=image_path.name,
                                style_analysis=style_analysis,
                                attribute_analysis=attribute_analysis,
                                selected_template=template,
                                selected_stylist=selected_stylist,
                                selected_coupon=selected_coupon,
                                processed_at=datetime.now()
                            )
                        except Exception as e:
                            self.logger.error(f"処理結果の作成に失敗しました: {str(e)}")
                        
                        # 結果をキャッシュに保存
                        if self.cache_manager:
                            self.cache_manager.set(cache_key, result)
                        
                        # 結果を追加
                        self.results.append(result)
                        self.logger.info(f"画像処理完了: {image_path.name}")
                        
                    except Exception as e:
                        self.logger.error(f"画像 {image_path.name} の処理中にエラーが発生しました: {str(e)}")
                    
                    # 進捗更新
                    processed_count += 1
                    tracker.update(processed_count, f"処理: {image_path.name}")
                    
                    # APIレート制限に対応するための遅延
                    await asyncio.sleep(self.api_delay)
                
                # バッチ間の遅延
                if batch_index < len(batches) - 1:
                    await asyncio.sleep(self.api_delay * 2)
        
        # 最終進捗更新
        tracker.update(total_images, total_images, f"処理完了: {len(self.results)}/{total_images}枚")
        
        self.logger.info(f"外部データを使用した複数画像処理完了: {len(self.results)}/{total_images}枚")
        return self.results
    
    def export_to_excel(self, output_path: Path) -> Path:
        """
        処理結果をExcelファイルに出力します。
        
        Args:
            output_path: 出力ファイルのパス
            
        Returns:
            エクスポートされたファイルのパス
            
        Raises:
            ExcelExportError: Excel出力処理でエラーが発生した場合
        """
        self.logger.info(f"Excel出力開始: 結果数={len(self.results)}, 出力先={output_path}")
        
        if not self.results:
            self.logger.warning("出力する結果がありません")
            raise ValidationError("出力する結果がありません")
        
        # Excel出力
        return self.excel_exporter.export(self.results, output_path)
    
    def export_to_text(self, output_path: Path) -> Path:
        """
        処理結果をテキストファイルに出力します。
        
        Args:
            output_path: 出力ファイルのパス
            
        Returns:
            エクスポートされたファイルのパス
            
        Raises:
            TextExportError: テキスト出力処理でエラーが発生した場合
        """
        self.logger.info(f"テキスト出力開始: 結果数={len(self.results)}, 出力先={output_path}")
        
        if not self.results:
            self.logger.warning("出力する結果がありません")
            raise ValidationError("出力する結果がありません")
        
        # テキスト出力
        return self.text_exporter.export(self.results, output_path)
    
    def get_excel_binary(self) -> bytes:
        """
        処理結果のExcelバイナリデータを取得します。
        
        Returns:
            Excelバイナリデータ
            
        Raises:
            ExcelExportError: Excel出力処理でエラーが発生した場合
        """
        self.logger.info(f"Excelバイナリデータ生成開始: 結果数={len(self.results)}")
        
        if not self.results:
            self.logger.warning("出力する結果がありません")
            raise ValidationError("出力する結果がありません")
        
        # Excelバイナリデータの取得
        return self.excel_exporter.get_binary_data(self.results)
    
    def get_text_content(self) -> str:
        """
        処理結果のテキストデータを取得します。
        
        Returns:
            テキストデータ
            
        Raises:
            TextExportError: テキスト出力処理でエラーが発生した場合
        """
        self.logger.info(f"テキストデータ生成開始: 結果数={len(self.results)}")
        
        if not self.results:
            self.logger.warning("出力する結果がありません")
            raise ValidationError("出力する結果がありません")
        
        # テキストデータの取得
        return self.text_exporter.get_text_content(self.results)
    
    def get_results(self) -> List[ProcessResultProtocol]:
        """
        処理結果のリストを取得します。
        
        Returns:
            処理結果のリスト
        """
        return self.results
    
    def clear_results(self) -> None:
        """
        処理結果をクリアします。
        """
        self.results = []
        self.logger.info("処理結果をクリアしました")
    
    def set_use_cache(self, use_cache: bool) -> None:
        """
        キャッシュを使用するかどうかを設定します。
        
        Args:
            use_cache: キャッシュを使用するかどうか
        """
        self.use_cache = use_cache
        self.image_analyzer.use_cache = use_cache
        self.logger.info(f"キャッシュ使用設定を変更しました: {use_cache}")
    
    async def retry_failed_images(self, image_paths: List[Path], use_cache: Optional[bool] = None) -> List[ProcessResultProtocol]:
        """
        失敗した画像を再処理します。
        
        Args:
            image_paths: 再処理する画像のパスリスト
            use_cache: キャッシュを使用するかどうか（Noneの場合はインスタンスの設定を使用）
            
        Returns:
            処理結果のリスト
        """
        self.logger.info(f"失敗画像の再処理開始: {len(image_paths)}枚")
        
        # キャッシュを使用するかどうかの判定
        should_use_cache = self.use_cache if use_cache is None else use_cache
        
        # 画像を再処理
        return await self.process_images(image_paths, use_cache=should_use_cache)
