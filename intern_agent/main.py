import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

VACANCIES_PATH = Path("vacancies.csv")
CRITERIA_PATH = Path("criteria.md")
REPORT_PATH = Path("report.md")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_N = 5
MAX_AGE_DAYS = 90

REMOTE_KEYWORDS = ("remote", "удален", "удалён")
SKILL_KEYWORDS = ("python", "c++", "алгоритм", "cpp")
DEFAULT_CRITERIA = (
    "Я студент, ищу стажировку или junior-вакансию. "
    "Знаю математику, C++, базовый Python, алгоритмы. Интересует удаленка."
)


def is_remote(location: object) -> bool:
    text = str(location).lower()
    return any(keyword in text for keyword in REMOTE_KEYWORDS)


def matches_skills(requirements: object) -> bool:
    text = str(requirements).lower()
    return any(keyword in text for keyword in SKILL_KEYWORDS)


def log_filter_step(stage: str, before: int, after: int) -> None:
    removed = before - after
    logger.info("%s: удалено %d, осталось %d", stage, removed, after)


def load_vacancies(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Не удалось прочитать CSV: %s", exc)
        raise ValueError(f"Битый или нечитаемый CSV: {exc}") from exc

    initial_count = len(df)
    logger.info("Загружено строк из CSV: %d", initial_count)

    df = df.dropna(how="all")
    log_filter_step("Удаление полностью пустых строк", initial_count, len(df))

    before = len(df)
    df = df.dropna(subset=["id", "title"])
    log_filter_step("Удаление строк без id/title", before, len(df))

    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    log_filter_step("Удаление дублей по id", before, len(df))

    before = len(df)
    df = df[df["is_junior"] == True]  # noqa: E712
    log_filter_step("Фильтр junior/стажировка (is_junior=True)", before, len(df))

    before = len(df)
    df = df[df["location"].apply(is_remote)]
    log_filter_step("Фильтр удалёнки", before, len(df))

    before = len(df)
    df = df[df["requirements"].apply(matches_skills)]
    log_filter_step("Фильтр навыков (Python/C++/алгоритмы)", before, len(df))

    if "published_date" in df.columns:
        before = len(df)
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=MAX_AGE_DAYS)
        df = df[df["published_date"] >= cutoff]
        log_filter_step(
            f"Фильтр свежести (не старше {MAX_AGE_DAYS} дней)", before, len(df)
        )

    filtered_out = initial_count - len(df)
    logger.info("Итого отфильтровано строк: %d", filtered_out)
    logger.info("Осталось вакансий для анализа: %d", len(df))

    return df.reset_index(drop=True)


def load_criteria(path: Path) -> str:
    if not path.exists():
        logger.warning("Файл %s не найден, используются критерии по умолчанию", path)
        return DEFAULT_CRITERIA

    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("Не удалось прочитать %s: %s", path, exc)
        return DEFAULT_CRITERIA

    if not text:
        logger.warning("Файл %s пуст, используются критерии по умолчанию", path)
        return DEFAULT_CRITERIA

    return text


def build_llm_prompt(vacancies: pd.DataFrame, criteria: str) -> str:
    vacancies_text = vacancies.to_string(index=False)
    return (
        "Ты помощник по подбору вакансий.\n\n"
        "Критерии кандидата:\n"
        f"{criteria}\n\n"
        "Доступные вакансии (уже отфильтрованы по junior, удалёнке, навыкам и дате):\n"
        f"{vacancies_text}\n\n"
        f"Выбери Топ-{TOP_N} наиболее подходящих вакансий из списка.\n"
        "Для каждой вакансии укажи:\n"
        "1. Почему подходит кандидату.\n"
        "2. Ключевые требования.\n"
        "3. Что может смущать или какой риск есть.\n\n"
        "Верни результат в Markdown:\n"
        f"- заголовок h1 «Топ-{TOP_N} вакансий»;\n"
        "- нумерованный список;\n"
        "- для каждой позиции подзаголовок h2 с названием и компанией;\n"
        "- три подпункта: «Почему подходит», «Требования», «Что смущает»."
    )


def ask_llm_for_top_vacancies(vacancies: pd.DataFrame, criteria: str) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise ValueError("Укажите действительный GROQ_API_KEY в файле .env")
    if not api_key.startswith("gsk_"):
        raise ValueError(
            "GROQ_API_KEY должен начинаться с gsk_. "
            "Получить ключ: https://console.groq.com/keys"
        )

    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    prompt = build_llm_prompt(vacancies, criteria)
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Ты эксперт по карьерному консультированию в IT.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
    except Exception as exc:
        error_name = type(exc).__name__
        if "AuthenticationError" in error_name:
            logger.error("Неверный API-ключ. Проверьте GROQ_API_KEY в .env")
        elif "RateLimitError" in error_name:
            logger.error(
                "Превышен лимит Groq (429). "
                "Подробнее: https://console.groq.com/docs/rate-limits"
            )
        else:
            logger.error("Ошибка при обращении к LLM: %s", exc)
        raise

    return response.choices[0].message.content.strip()


def build_fallback_report(vacancies: pd.DataFrame, error: str) -> str:
    logger.warning("Используется fallback-отчёт без LLM: %s", error)

    rows = vacancies.head(TOP_N)
    lines = [
        f"# Топ-{TOP_N} вакансий (fallback без LLM)",
        "",
        f"_LLM недоступен: {error}_",
        "",
        "Ранжирование выполнено по правилам: junior + удалёнка + навыки + свежесть.",
        "",
    ]

    for index, row in rows.iterrows():
        lines.extend(
            [
                f"## {index + 1}. {row['title']} — {row['company']}",
                "",
                "**Почему подходит:** вакансия прошла rule-based фильтрацию по уровню, "
                "формату работы и ключевым навыкам.",
                "",
                f"**Требования:** {row['requirements']}",
                "",
                f"**Что смущает:** зарплата {row['salary']}, локация {row['location']}. "
                "Детальный разбор недоступен без LLM.",
                "",
            ]
        )

    return "\n".join(lines).strip()


def save_report(content: str, path: Path) -> None:
    path.write_text(content, encoding="utf-8")
    logger.info("Отчет сохранен в %s", path)


def main() -> None:
    load_dotenv(override=True)
    logger.info("Запуск агента фильтрации вакансий")

    try:
        vacancies = load_vacancies(VACANCIES_PATH)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Ошибка загрузки вакансий: %s", exc)
        save_report(
            f"# Ошибка\n\nНе удалось обработать входные данные: {exc}",
            REPORT_PATH,
        )
        return

    criteria = load_criteria(CRITERIA_PATH)
    logger.info("Критерии загружены (%d символов)", len(criteria))

    if vacancies.empty:
        logger.warning("После фильтрации не осталось вакансий для анализа")
        save_report(
            "# Результат\n\nНет подходящих вакансий после rule-based фильтрации.",
            REPORT_PATH,
        )
        return

    try:
        report = ask_llm_for_top_vacancies(vacancies, criteria)
        logger.info("LLM успешно сформировал отчёт")
    except Exception as exc:
        report = build_fallback_report(vacancies, str(exc))

    save_report(report, REPORT_PATH)
    logger.info("Работа агента завершена")


if __name__ == "__main__":
    main()
