"""
インターフェース定義モジュール

このモジュールでは、アプリケーション全体で使用するインターフェースを定義します。
Python 3.8以降の typing.Protocol を使用して、構造的サブタイピングをサポートします。
"""

from typing import Protocol, Dict, List, Optional, Any, TypeVar, Generic
from datetime import datetime
from pathlib import Path


# 基本的なキャッシュエントリのインターフェース
class CacheEntryProtocol(Protocol):
    """キャッシュエントリのインターフェース"""
    data: Any
    timestamp: float
    ttl: Optional[float]


# キャッシュマネージャーのインターフェース
class CacheManagerProtocol(Protocol):
    """キャッシュマネージャーのインターフェース"""
    
    def get(self, key: str, context: str = "") -> Optional[Any]:
        """
        指定されたキーのキャッシュデータを取得します。
        
        Args:
            key: キャッシュキー
            context: キャッシュコンテキスト（オプション）
            
        Returns:
            キャッシュデータ、または存在しない場合はNone
        """
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, context: str = "") -> None:
        """
        キャッシュにデータを設定します。
        
        Args:
            key: キャッシュキー
            value: キャッシュするデータ
            ttl: 有効期限（秒）（オプション）
            context: キャッシュコンテキスト（オプション）
        """
        ...
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        キャッシュをクリアします。
        
        Args:
            pattern: クリアするキーのパターン（オプション）
            
        Returns:
            クリアされたエントリ数
        """
        ...


# 画像分析結果のインターフェース
class StyleAnalysisProtocol(Protocol):
    """画像分析結果のインターフェース"""
    category: str
    features: Dict[str, str]
    keywords: List[str]


# 属性分析結果のインターフェース
class AttributeAnalysisProtocol(Protocol):
    """属性分析結果のインターフェース"""
    sex: str
    length: str


# テンプレートのインターフェース
class TemplateProtocol(Protocol):
    """テンプレートのインターフェース"""
    category: str
    title: str
    menu: str
    comment: str
    hashtag: str


# テンプレートマネージャーのインターフェース
class TemplateManagerProtocol(Protocol):
    """テンプレートマネージャーのインターフェース"""
    
    def get_templates_by_category(self, category: str) -> List[TemplateProtocol]:
        """
        指定されたカテゴリのテンプレートリストを取得します。
        
        Args:
            category: カテゴリ名
            
        Returns:
            テンプレートリスト
        """
        ...
    
    def get_all_categories(self) -> List[str]:
        """
        全てのカテゴリリストを取得します。
        
        Returns:
            カテゴリリスト
        """
        ...
    
    def find_best_template(self, analysis: StyleAnalysisProtocol) -> Optional[TemplateProtocol]:
        """
        分析結果に最も合うテンプレートを検索します。
        
        Args:
            analysis: 分析結果
            
        Returns:
            最適なテンプレート、または見つからない場合はNone
        """
        ...


# スタイリスト情報のインターフェース
class StylistInfoProtocol(Protocol):
    """スタイリスト情報のインターフェース"""
    name: str
    description: str
    position: Optional[str]


# クーポン情報のインターフェース
class CouponInfoProtocol(Protocol):
    """クーポン情報のインターフェース"""
    name: str
    price: Optional[str]


# スクレイパーサービスのインターフェース
class ScraperServiceProtocol(Protocol):
    """スクレイパーサービスのインターフェース"""
    
    async def scrape_stylists(self, url: str) -> List[StylistInfoProtocol]:
        """
        スタイリスト情報をスクレイピングします。
        
        Args:
            url: スクレイピング対象のURL
            
        Returns:
            スタイリスト情報のリスト
        """
        ...
    
    async def scrape_coupons(self, url: str) -> List[CouponInfoProtocol]:
        """
        クーポン情報をスクレイピングします。
        
        Args:
            url: スクレイピング対象のURL
            
        Returns:
            クーポン情報のリスト
        """
        ...


# Gemini API サービスのインターフェース
class GeminiServiceProtocol(Protocol):
    """Gemini API サービスのインターフェース"""
    
    async def analyze_image(self, image_path: Path, categories: List[str]) -> Optional[StyleAnalysisProtocol]:
        """
        画像を分析します。
        
        Args:
            image_path: 画像ファイルのパス
            categories: カテゴリリスト
            
        Returns:
            分析結果、またはエラー時はNone
        """
        ...
    
    async def analyze_attributes(self, image_path: Path) -> Optional[AttributeAnalysisProtocol]:
        """
        画像の属性を分析します。
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            属性分析結果、またはエラー時はNone
        """
        ...
    
    async def select_stylist(self, 
                            image_path: Path, 
                            stylists: List[StylistInfoProtocol], 
                            analysis: StyleAnalysisProtocol) -> Optional[StylistInfoProtocol]:
        """
        画像に最適なスタイリストを選択します。
        
        Args:
            image_path: 画像ファイルのパス
            stylists: スタイリスト情報のリスト
            analysis: 画像分析結果
            
        Returns:
            選択されたスタイリスト情報、またはエラー時はNone
        """
        ...
    
    async def select_coupon(self, 
                           image_path: Path, 
                           coupons: List[CouponInfoProtocol], 
                           analysis: StyleAnalysisProtocol) -> Optional[CouponInfoProtocol]:
        """
        画像に最適なクーポンを選択します。
        
        Args:
            image_path: 画像ファイルのパス
            coupons: クーポン情報のリスト
            analysis: 画像分析結果
            
        Returns:
            選択されたクーポン情報、またはエラー時はNone
        """
        ...


# 処理結果のインターフェース
class ProcessResultProtocol(Protocol):
    """処理結果のインターフェース"""
    image_name: str
    style_analysis: StyleAnalysisProtocol
    attribute_analysis: AttributeAnalysisProtocol
    selected_template: TemplateProtocol
    alternative_templates: List[TemplateProtocol]
    user_selected_template: Optional[TemplateProtocol]
    selected_stylist: StylistInfoProtocol
    selected_coupon: CouponInfoProtocol
    processed_at: datetime


# メインプロセッサのインターフェース
class MainProcessorProtocol(Protocol):
    """メインプロセッサのインターフェース"""
    
    async def process_single_image(self, image_path: Path) -> Optional[ProcessResultProtocol]:
        """
        単一の画像を処理します。
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            処理結果、またはエラー時はNone
        """
        ...
    
    async def process_images(self, image_paths: List[Path]) -> List[ProcessResultProtocol]:
        """
        複数の画像を処理します。
        
        Args:
            image_paths: 画像ファイルのパスリスト
            
        Returns:
            処理結果のリスト
        """
        ...


# Excelエクスポーターのインターフェース
class ExcelExporterProtocol(Protocol):
    """Excelエクスポーターのインターフェース"""
    
    def export(self, results: List[ProcessResultProtocol], output_path: Path) -> Path:
        """
        処理結果をExcel形式でエクスポートします。
        
        Args:
            results: 処理結果のリスト
            output_path: 出力ファイルのパス
            
        Returns:
            エクスポートされたファイルのパス
        """
        ...
    
    def get_binary_data(self, results: List[ProcessResultProtocol]) -> bytes:
        """
        処理結果のExcelバイナリデータを取得します。
        
        Args:
            results: 処理結果のリスト
            
        Returns:
            Excelバイナリデータ
        """
        ...


# テキストエクスポーターのインターフェース
class TextExporterProtocol(Protocol):
    """テキストエクスポーターのインターフェース"""
    
    def export(self, results: List[ProcessResultProtocol], output_path: Path) -> Path:
        """
        処理結果をテキスト形式でエクスポートします。
        
        Args:
            results: 処理結果のリスト
            output_path: 出力ファイルのパス
            
        Returns:
            エクスポートされたファイルのパス
        """
        ...
    
    def get_text_content(self, results: List[ProcessResultProtocol]) -> str:
        """
        処理結果のテキストデータを取得します。
        
        Args:
            results: 処理結果のリスト
            
        Returns:
            テキストデータ
        """
        ...
