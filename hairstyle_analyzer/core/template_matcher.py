"""
テンプレートマッチングモジュール

このモジュールでは、画像分析結果に基づいて最適なテンプレートを選択するための機能を提供します。
カテゴリマッチング、特徴ベースのスコアリング、最適テンプレート選択ロジックが含まれます。
"""

import logging
import difflib
import random
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set

from ..data.models import Template
from ..data.interfaces import StyleAnalysisProtocol, TemplateManagerProtocol
from ..utils.errors import TemplateError, GeminiAPIError, with_error_handling
from ..services.gemini.gemini_service import GeminiService


class TemplateMatcher:
    """
    テンプレートマッチングクラス
    
    画像分析結果に基づいて最適なテンプレートを選択します。
    カテゴリマッチング、特徴ベースのスコアリング、キーワード類似度計算などの機能が含まれます。
    """
    
    def __init__(self, template_manager: TemplateManagerProtocol):
        """
        初期化
        
        Args:
            template_manager: テンプレートマネージャー
        """
        self.logger = logging.getLogger(__name__)
        self.template_manager = template_manager
    
    @with_error_handling(TemplateError, "テンプレートマッチング処理でエラーが発生しました")
    def find_best_template(self, analysis: StyleAnalysisProtocol) -> Optional[Template]:
        """
        分析結果に最適なテンプレートを検索します。
        
        Args:
            analysis: 画像分析結果
            
        Returns:
            最適なテンプレート、見つからない場合はNone
            
        Raises:
            TemplateError: テンプレート処理中にエラーが発生した場合
        """
        self.logger.info(f"テンプレートマッチング開始: カテゴリ={analysis.category}")
        
        # テンプレートマネージャーを使用して最適なテンプレートを検索
        template = self.template_manager.find_best_template(analysis)
        
        if template:
            self.logger.info(f"最適なテンプレートを見つけました: タイトル={template.title}")
        else:
            self.logger.warning(f"適切なテンプレートが見つかりませんでした")
            
        return template
    
    def find_alternative_templates(self, analysis: StyleAnalysisProtocol, count: int = 3) -> List[Template]:
        """
        代替テンプレートを検索します。
        
        Args:
            analysis: 画像分析結果
            count: 返す代替テンプレートの数
            
        Returns:
            代替テンプレートのリスト
        """
        self.logger.info(f"代替テンプレート検索開始: カテゴリ={analysis.category}")
        
        templates = []
        
        # まず分析結果のカテゴリに一致するテンプレートを取得
        category_templates = self.template_manager.get_templates_by_category(analysis.category)
        
        # 一致するカテゴリがない場合、全テンプレートを対象にする
        if not category_templates:
            category_templates = self.template_manager.get_all_templates()
        
        # 十分なテンプレートがある場合は、分析結果のキーワードや特徴に基づいてスコアリング
        if len(category_templates) > 1:
            # スコアリングロジックの実装
            scored_templates = self._score_templates(category_templates, analysis)
            
            # 結果をスコア降順にソート
            scored_templates.sort(key=lambda x: x[0], reverse=True)
            
            # 上位のテンプレートを返す（最良のテンプレートを除く）
            templates = [template for score, template in scored_templates[:count+1]]
            
            # 既に find_best_template で最良のテンプレートが返されている可能性があるため、
            # リストが十分な長さの場合は最初のテンプレートを除外
            if len(templates) > count:
                templates = templates[1:count+1]
        else:
            templates = category_templates[:count]
        
        return templates
    
    def _score_templates(self, templates: List[Template], analysis: StyleAnalysisProtocol) -> List[Tuple[float, Template]]:
        """
        テンプレートをスコアリングします。
        
        Args:
            templates: スコアリングするテンプレートのリスト
            analysis: 画像分析結果
            
        Returns:
            (スコア, テンプレート) のタプルのリスト
        """
        scored_templates = []
        
        # 分析結果からキーワードを取得
        analysis_keywords = set(analysis.keywords)
        
        for template in templates:
            score = 0.0
            
            # カテゴリの一致（最も重要）
            if template.category == analysis.category:
                score += 3.0
            
            # キーワードの一致（2番目に重要）
            template_tags = set(template.get_hashtags())
            if template_tags and analysis_keywords:
                # 共通するキーワードの割合
                common_keywords = template_tags.intersection(analysis_keywords)
                keyword_score = len(common_keywords) / max(len(template_tags), 1) * 2.0
                score += keyword_score
            
            # テキスト内容の類似度（3番目に重要）
            # タイトル、コメント、メニューなどのテキスト内容に分析結果の特徴が含まれているかを評価
            combined_text = f"{template.title} {template.comment} {template.menu}".lower()
            
            # 特徴値のリスト
            features_values = [
                analysis.features.color,
                analysis.features.cut_technique,
                analysis.features.styling,
                analysis.features.impression
            ]
            
            # 各特徴についてテキスト内での一致を評価
            for feature_value in features_values:
                if feature_value.lower() in combined_text:
                    score += 0.5
            
            scored_templates.append((score, template))
        
        return scored_templates
    
    def get_template_by_category(self, category: str) -> Optional[Template]:
        """
        指定されたカテゴリから最初のテンプレートを取得します。
        
        Args:
            category: カテゴリ名
            
        Returns:
            テンプレート、見つからない場合はNone
        """
        templates = self.template_manager.get_templates_by_category(category)
        if templates:
            return templates[0]
        return None
    
    def get_random_template(self) -> Optional[Template]:
        """
        ランダムなテンプレートを取得します。
        
        Returns:
            テンプレート、見つからない場合はNone
        """
        import random
        templates = self.template_manager.get_all_templates()
        if templates:
            return random.choice(templates)
        return None

    async def find_best_template_with_ai(
        self,
        image_path: Path,
        gemini_service: GeminiService,
        analysis: Optional[StyleAnalysisProtocol] = None,
        use_category_filter: bool = True,
        max_templates: int = 50
    ) -> Tuple[Optional[Template], List[Template], Optional[str], bool]:
        """
        Gemini AIを使って最適なテンプレートとその代替候補を選択します。
        
        Args:
            image_path: 画像ファイルのパス
            gemini_service: Gemini APIサービスインスタンス
            analysis: 画像分析結果（オプション、指定がない場合はAI選択のみで実行）
            use_category_filter: カテゴリでテンプレートをフィルタするかどうか
            max_templates: AIに送信する最大テンプレート数
            
        Returns:
            (選択されたテンプレート, 代替テンプレートリスト, 選択理由, 成功したかどうか)のタプル
        """
        self.logger.info(f"AIによるテンプレートマッチング開始: 画像={image_path.name}")
        
        # 利用可能なすべてのカテゴリを取得
        available_categories = self.template_manager.get_all_categories()
        
        # 最適なカテゴリの決定
        selected_category = None
        
        if analysis and use_category_filter:
            # 既存の分析結果からカテゴリを取得
            selected_category = analysis.category
            self.logger.info(f"分析結果から取得したカテゴリ: {selected_category}")
        
        # カテゴリが不明または利用可能なカテゴリに含まれていない場合、AIに選んでもらう
        if not selected_category or selected_category not in available_categories:
            try:
                selected_category = await gemini_service.get_matching_category(image_path, available_categories)
                self.logger.info(f"AIが選択したカテゴリ: {selected_category}")
            except Exception as e:
                self.logger.error(f"カテゴリ選択中にエラーが発生しました: {str(e)}")
                
                # エラー時はデフォルトカテゴリまたは最初のカテゴリを使用
                if available_categories:
                    selected_category = available_categories[0]
                    self.logger.warning(f"エラーによりデフォルトカテゴリを使用します: {selected_category}")
                else:
                    self.logger.error("利用可能なカテゴリがありません")
                    return None, [], "利用可能なカテゴリがありません", False
        
        # 選択されたカテゴリのテンプレートを取得
        templates = self.template_manager.get_templates_by_category(selected_category)
        self.logger.info(f"カテゴリ '{selected_category}' で検索: {len(templates)}件のテンプレート")
        
        # カテゴリに一致するテンプレートがない場合は全テンプレートを使用
        if not templates:
            self.logger.warning(f"カテゴリ '{selected_category}' に一致するテンプレートがありません。全テンプレートを使用します。")
            templates = self.template_manager.get_all_templates()
            self.logger.info(f"全テンプレートを使用: {len(templates)}件")
        
        if not templates:
            self.logger.error("テンプレートが見つかりません")
            return None, [], "テンプレートが見つかりません", False
        
        # GeminiServiceを使用してテンプレート選択
        try:
            template_infos, success = await gemini_service.select_best_template(
                image_path=image_path,
                templates=templates,
                analysis=analysis,
                category_filter=use_category_filter
            )
            
            if not template_infos or len(template_infos) == 0:
                self.logger.warning("テンプレート情報が空です")
                return None, [], "テンプレート選択結果が空です", False
                
            # 主要なテンプレートと選択理由
            primary_template_id, primary_reason = template_infos[0]
            primary_template = templates[primary_template_id]
            
            # 代替テンプレート
            alternative_templates = []
            for template_id, _ in template_infos[1:]:
                if 0 <= template_id < len(templates):
                    alternative_templates.append(templates[template_id])
            
            self.logger.info(f"AIがテンプレートを選択しました: 主要={primary_template.title}, 代替={len(alternative_templates)}個")
            return primary_template, alternative_templates, primary_reason, True
            
        except ValueError as e:
            # テンプレートリストが空など、値関連のエラー
            self.logger.error(f"値エラーが発生しました: {str(e)}")
            return None, [], f"テンプレート選択エラー: {str(e)}", False
            
        except GeminiAPIError as e:
            # GeminiAPI関連のエラー
            self.logger.error(f"GeminiAPIエラーが発生しました: {str(e)}")
            return None, [], f"GeminiAPIエラー: {str(e)}", False
            
        except Exception as e:
            # その他の予期しないエラー
            self.logger.error(f"AIによるテンプレート選択中に予期しないエラーが発生しました: {str(e)}")
            return None, [], f"予期しないエラー: {str(e)}", False
