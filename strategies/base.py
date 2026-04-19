"""
strategies/base.py — 策略插件抽象基类

每个策略封装：
  1. price_structure_defaults: 预计算字段的默认值
  2. build_price_structure(df): 从 OHLCV 计算所有策略字段，含 ema20 子 dict
  3. build_features(market_data, lang, klines_completed): 组装 AI prompt 特征块
  4. get_skip_struct(): 已在特征块的字段（从原始数据中剥离）
  5. rate_bar(bar, prev_bar): 信号棒评级
  6. 12 个 prompt 属性（6 中文段 + 6 英文段）
"""

from __future__ import annotations
import abc
import pandas as pd


class BaseStrategy(abc.ABC):
    """
    策略插件抽象基类。

    子类必须实现所有抽象方法/属性。

    price_structure_defaults 使用类级常量缓存，property 每次返回同一对象，
    避免模块级别名失效（data_feed.PRICE_STRUCTURE_DEFAULTS = strategy.price_structure_defaults）。
    """

    # ── 标识 ──────────────────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def strategy_id(self) -> str:
        """唯一小写标识符，如 'brooks'。"""

    # ── 默认值 ────────────────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def price_structure_defaults(self) -> dict:
        """
        策略预计算字段的默认值字典。含 'ema20' key（默认 {}）。
        必须返回缓存的类级常量，不每次新建（避免模块级别名失效）。
        调用方读取列表类型字段时须自行复制：list(v)。
        """

    # ── 价格结构计算 ───────────────────────────────────────────────────────────

    @abc.abstractmethod
    def build_price_structure(self, df: pd.DataFrame) -> dict:
        """
        计算所有策略预计算字段，包含 'ema20' 子 dict。

        Parameters
        ----------
        df : pd.DataFrame
            250 根 OHLCV DataFrame，列：datetime, open, high, low, close, volume。
            已由 fetch() 做过 dropna(subset=['close']) 和 tail(250)。

        Returns
        -------
        dict
            包含所有 price_structure_defaults 中的 key，以及 'ema20' 子 dict。
            df 长度 < 20 时返回 defaults 拷贝（含空 ema20 = {}）。
            纯函数：无 I/O，无全局状态写入，不调用任何 tqsdk 函数。
        """

    # ── 特征块 ─────────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def build_features(
        self,
        market_data: dict,
        lang: str,
        klines_completed: list[dict] | None = None,
    ) -> dict:
        """
        构建注入 AI prompt 的预计算特征块（JSON 序列化后发送）。

        Parameters
        ----------
        market_data : dict
            data_feed.fetch(symbol) 的完整返回值，含 ema20 子 dict、quote 子 dict
            以及所有 price_structure 字段（顶层）。
        lang : str
            'zh' 或 'en'。'en' 时所有 key 和枚举值须翻译。
        klines_completed : list[dict] | None
            已完成 K 线列表（由 _build_user_message 传入，避免重复计算）。
            None 时由实现方内部计算。

        Returns
        -------
        dict
            序列化前的特征字典。
            必须包含 '_last_bar_rating' 私有 key（中文评级字符串），
            调用方 (_build_user_message) 在序列化前 pop() 取出。
        """

    # ── 跳过集合 ───────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_skip_struct(self) -> frozenset[str]:
        """
        已在特征块中表示的字段名，从原始 market_data JSON 中剥离（避免重复）。
        对应 analyzers.py 原 _SKIP_STRUCT 集合。

        规则：
        - 包含 price_structure_defaults 的所有 key（含 _ema 调试字段）
        - 包含 'ema20'（EMA 信息已在特征块给出）
        - 包含 'is_limit_up', 'is_limit_down'（在特征块中展示）
        - 不包含通用的 _SKIP_FIELDS（pending, error 等），那些由 analyzers 层管理
        """

    # ── 信号棒评级 ─────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def rate_bar(self, bar: dict, prev_bar: dict | None = None) -> str:
        """
        对单根已完成 K 线评级。

        Parameters
        ----------
        bar : dict
            含 open, high, low, close, body_ratio, close_position 的 K 线 dict。
        prev_bar : dict | None
            前一根 K 线（用于相对强度检验：该棒是否拓展了方向性极值）。
            None 时不做相对检验。

        Returns
        -------
        str
            中文标准值之一：
            "强多头棒" / "中多头棒" / "弱多头棒" /
            "强空头棒" / "中空头棒" / "弱空头棒" / "十字星"
        """

    # ── System Prompt 抽象属性（12 个：6 中文 + 6 英文）─────────────────────────

    @property
    @abc.abstractmethod
    def prompt_head_zh(self) -> str:
        """ZH: 图表设置 + 第一步（市场状态）"""

    @property
    @abc.abstractmethod
    def prompt_entry_block_zh(self) -> str:
        """ZH: 第二步~第五步：设置识别、信号棒、入场理由、SL/TP"""

    @property
    @abc.abstractmethod
    def prompt_trail_mgmt_zh(self) -> str:
        """ZH: 动态止损管理（持仓状态专用）"""

    @property
    @abc.abstractmethod
    def prompt_decision_no_pos_zh(self) -> str:
        """ZH: 无持仓决策规则"""

    @property
    @abc.abstractmethod
    def prompt_decision_has_pos_zh(self) -> str:
        """ZH: 有持仓决策规则"""

    @property
    @abc.abstractmethod
    def prompt_output_zh(self) -> str:
        """ZH: 输出格式要求和 JSON schema"""

    @property
    @abc.abstractmethod
    def prompt_head_en(self) -> str:
        """EN mirror of prompt_head_zh"""

    @property
    @abc.abstractmethod
    def prompt_entry_block_en(self) -> str:
        """EN mirror of prompt_entry_block_zh"""

    @property
    @abc.abstractmethod
    def prompt_trail_mgmt_en(self) -> str:
        """EN mirror of prompt_trail_mgmt_zh"""

    @property
    @abc.abstractmethod
    def prompt_decision_no_pos_en(self) -> str:
        """EN mirror of prompt_decision_no_pos_zh"""

    @property
    @abc.abstractmethod
    def prompt_decision_has_pos_en(self) -> str:
        """EN mirror of prompt_decision_has_pos_zh"""

    @property
    @abc.abstractmethod
    def prompt_output_en(self) -> str:
        """EN mirror of prompt_output_zh"""

    # ── Composite Prompt 默认实现 ─────────────────────────────────────────────

    def get_default_system_prompt(self, lang: str = "zh") -> str:
        """无持仓 prompt：head + entry_block + decision_no_pos + output"""
        if lang == "en":
            return (self.prompt_head_en + self.prompt_entry_block_en
                    + self.prompt_decision_no_pos_en + self.prompt_output_en)
        return (self.prompt_head_zh + self.prompt_entry_block_zh
                + self.prompt_decision_no_pos_zh + self.prompt_output_zh)

    def get_holding_system_prompt(self, lang: str = "zh") -> str:
        """有持仓 prompt：head + trail_mgmt + decision_has_pos + output"""
        if lang == "en":
            return (self.prompt_head_en + self.prompt_trail_mgmt_en
                    + self.prompt_decision_has_pos_en + self.prompt_output_en)
        return (self.prompt_head_zh + self.prompt_trail_mgmt_zh
                + self.prompt_decision_has_pos_zh + self.prompt_output_zh)

    def get_merged_system_prompt(self, lang: str = "zh") -> str:
        """全段合并 prompt（qwen3 前缀缓存用）：6 段全部拼接"""
        if lang == "en":
            return (self.prompt_head_en + self.prompt_entry_block_en
                    + self.prompt_trail_mgmt_en + self.prompt_decision_no_pos_en
                    + self.prompt_decision_has_pos_en + self.prompt_output_en)
        return (self.prompt_head_zh + self.prompt_entry_block_zh
                + self.prompt_trail_mgmt_zh + self.prompt_decision_no_pos_zh
                + self.prompt_decision_has_pos_zh + self.prompt_output_zh)

    # ── 可选扩展（非抽象，子类按需覆盖）────────────────────────────────────

    @property
    def use_trailing_stop(self) -> bool:
        """是否启用追踪止损守卫。默认 True；RetailStrategy 覆盖为 False。"""
        return True

    def build_user_context(
        self,
        history: list[dict],
        market_data: dict,
        positions: list[dict] | None,
        lang: str,
    ) -> str:
        """可选：返回注入用户消息末尾的额外上下文。默认空字符串（Brooks 不使用）。"""
        return ""

    def translate_output(self, raw: dict, market_data: dict) -> "dict | None":
        """可选：将策略专属输出格式翻译为标准 analysis dict。返回 None 表示无需翻译。"""
        return None
