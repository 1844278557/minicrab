"""受限记忆文件操作：只对 USER.md 和 MEMORY.md 做单条增改删。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MemoryOperationResult:
    """一次记忆操作的执行结果。"""

    changed: bool
    message: str


class MemoryOperationApplicator:
    """把 append/replace/delete 操作安全应用到受限记忆文件。"""

    _USER_TEMPLATE = "# User Memory\n\n## Preferences\n\n"
    _MEMORY_TEMPLATE = (
        "# Working Memory\n\n"
        "## Business Rules\n\n"
        "## Project Decisions\n\n"
        "## Lessons Learned\n\n"
        "## Important External Information\n\n"
    )

    def __init__(self, workspace: Path):
        """记录允许写入的记忆文件路径。"""
        self.workspace = workspace
        self.user_file = workspace / "USER.md"
        self.memory_file = workspace / "memory" / "MEMORY.md"

    def apply(
        self,
        *,
        target: str,
        op: str,
        section: str | None = None,
        match: str | None = None,
        content: str | None = None,
    ) -> MemoryOperationResult:
        """执行一条受限记忆操作。"""
        normalized_target = (target or "").lower()
        normalized_op = (op or "").lower()
        if normalized_target not in {"user", "memory"}:
            return MemoryOperationResult(False, "Error: target must be 'user' or 'memory'.")
        if normalized_op not in {"append", "replace", "delete"}:
            return MemoryOperationResult(False, "Error: op must be append, replace, or delete.")

        path = self._target_file(normalized_target)
        text = self._read_or_template(path, normalized_target)
        if normalized_op == "append":
            result_text = self._append(text, normalized_target, section, content)
        elif normalized_op == "replace":
            result_text = self._replace(text, match, content)
        else:
            result_text = self._delete(text, match)

        if isinstance(result_text, MemoryOperationResult):
            return result_text
        if result_text == text:
            return MemoryOperationResult(False, "Memory was unchanged.")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result_text.rstrip() + "\n", encoding="utf-8")
        return MemoryOperationResult(True, f"{path.name} updated.")

    def _target_file(self, target: str) -> Path:
        """返回目标记忆文件。"""
        return self.user_file if target == "user" else self.memory_file

    def _read_or_template(self, path: Path, target: str) -> str:
        """读取现有文件，不存在时使用最小模板。"""
        if path.exists():
            return path.read_text(encoding="utf-8")
        return self._USER_TEMPLATE if target == "user" else self._MEMORY_TEMPLATE

    def _append(
        self,
        text: str,
        target: str,
        section: str | None,
        content: str | None,
    ) -> str | MemoryOperationResult:
        """追加一条记忆到指定 section。"""
        bullet = self._normalize_single_bullet(content)
        if isinstance(bullet, MemoryOperationResult):
            return bullet
        if self._contains_exact_bullet(text, bullet):
            return MemoryOperationResult(False, "Memory already exists.")

        target_section = (section or self._default_section(target)).strip()
        if not target_section:
            return MemoryOperationResult(False, "Error: section is required for append.")
        return self._append_under_section(text, target_section, bullet)

    def _replace(
        self,
        text: str,
        match: str | None,
        content: str | None,
    ) -> str | MemoryOperationResult:
        """精确替换一条记忆。"""
        old = self._normalize_match(match)
        if isinstance(old, MemoryOperationResult):
            return old
        new = self._normalize_single_bullet(content)
        if isinstance(new, MemoryOperationResult):
            return new

        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.strip() == old:
                if line == new:
                    return MemoryOperationResult(False, "Memory was unchanged.")
                lines[index] = new
                return "\n".join(lines)
        return MemoryOperationResult(False, "No exact matching memory found.")

    def _delete(self, text: str, match: str | None) -> str | MemoryOperationResult:
        """精确删除一条记忆。"""
        old = self._normalize_match(match)
        if isinstance(old, MemoryOperationResult):
            return old

        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.strip() == old:
                del lines[index]
                return "\n".join(lines)
        return MemoryOperationResult(False, "No exact matching memory found.")

    def _append_under_section(self, text: str, section: str, bullet: str) -> str:
        """把 bullet 插入到指定 Markdown section；section 不存在时创建。"""
        lines = text.splitlines()
        heading_index = self._find_heading(lines, section)
        if heading_index is None:
            insert_at = self._footer_index(lines)
            block = self._new_section_block(lines, insert_at, section, bullet)
            lines[insert_at:insert_at] = block
            return "\n".join(lines)

        next_heading = self._next_heading(lines, heading_index + 1)
        insert_at = next_heading
        while insert_at > heading_index + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        block = [bullet]
        if insert_at < len(lines) and lines[insert_at].startswith("#"):
            block.append("")
        lines[insert_at:insert_at] = block
        return "\n".join(lines)

    def _new_section_block(self, lines: list[str], insert_at: int, section: str, bullet: str) -> list[str]:
        """生成缺失 section 的插入块。"""
        block: list[str] = []
        if insert_at > 0 and lines[insert_at - 1].strip():
            block.append("")
        block.extend([f"## {section}", "", bullet, ""])
        return block

    @staticmethod
    def _normalize_single_bullet(content: str | None) -> str | MemoryOperationResult:
        """校验并规范化单条 Markdown bullet。"""
        raw = (content or "").strip()
        if not raw:
            return MemoryOperationResult(False, "Error: content is required.")
        if "\n" in raw or "\r" in raw:
            return MemoryOperationResult(False, "Error: content must be one memory bullet.")
        if not raw.startswith("- "):
            return MemoryOperationResult(False, "Error: content must start with '- '.")
        return raw

    @staticmethod
    def _normalize_match(match: str | None) -> str | MemoryOperationResult:
        """校验精确匹配文本。"""
        raw = (match or "").strip()
        if not raw:
            return MemoryOperationResult(False, "Error: match is required.")
        if "\n" in raw or "\r" in raw:
            return MemoryOperationResult(False, "Error: match must be one memory bullet.")
        return raw

    @staticmethod
    def _contains_exact_bullet(text: str, bullet: str) -> bool:
        """判断文件中是否已有完全相同的 bullet。"""
        return any(line.strip() == bullet for line in text.splitlines())

    @staticmethod
    def _default_section(target: str) -> str:
        """给没有指定 section 的追加操作提供默认位置。"""
        return "Preferences" if target == "user" else "Project Decisions"

    @staticmethod
    def _find_heading(lines: list[str], section: str) -> int | None:
        """按标题文本查找任意级别 Markdown heading。"""
        expected = section.strip().lower()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            title = stripped.lstrip("#").strip().lower()
            if title == expected:
                return index
        return None

    @staticmethod
    def _next_heading(lines: list[str], start: int) -> int:
        """查找下一个 Markdown heading。"""
        for index in range(start, len(lines)):
            if lines[index].startswith("#"):
                return index
        return len(lines)

    @staticmethod
    def _footer_index(lines: list[str]) -> int:
        """把新 section 插在页脚分隔线前。"""
        for index, line in enumerate(lines):
            if line.strip() == "---":
                return index
        return len(lines)
