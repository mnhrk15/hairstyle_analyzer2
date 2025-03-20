#!/usr/bin/env python
"""
ヘアスタイル分析アプリのUIテストランナー

このスクリプトはヘアスタイル分析アプリのUIテストを実行します。
"""

import os
import sys
import pytest
from pathlib import Path

# プロジェクトルートディレクトリをPythonパスに追加
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """UIテストを実行する関数"""
    # テスト対象ディレクトリ
    test_dir = current_dir / "unit" / "ui"
    
    # テストディレクトリの存在確認
    if not test_dir.exists():
        print(f"エラー: テストディレクトリが見つかりません: {test_dir}")
        return 1
    
    # テスト実行
    print(f"UIテストを実行します: {test_dir}")
    return pytest.main(["-v", str(test_dir)])

if __name__ == "__main__":
    # 環境変数の設定（必要に応じて）
    os.environ["PYTHONPATH"] = str(project_root)
    
    # テスト実行
    result = run_tests()
    sys.exit(result) 