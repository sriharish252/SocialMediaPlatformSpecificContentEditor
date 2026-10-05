import asyncio

import streamlit as st
from dotenv import load_dotenv

from social_editor.config import make_llm, missing_key_error, model_name
from social_editor.crew import run_platforms
from social_editor.platforms import load_platforms

load_dotenv()

st.set_page_config(page_title="Social Media Content Editor", page_icon="✍️")
st.title("Social Media Content Editor")
st.caption(
    "One post in, one per platform out. For each platform an editor agent drafts, "
    "a critic reviews and the editor rewrites. Platforms run in parallel."
)

model = model_name()
if error := missing_key_error(model):
    st.error(error, icon="🔑")
    st.stop()

platforms = load_platforms()

with st.form("editor"):
    content = st.text_area(
        "Your content",
        height=180,
        placeholder="Calling all creative minds! The MetalBoys are getting a brand new logo...",
    )
    chosen = st.pills(
        "Platforms",
        options=list(platforms),
        format_func=lambda key: platforms[key].name,
        selection_mode="multi",
        default=list(platforms),
    )
    submitted = st.form_submit_button("Rewrite", type="primary")

if submitted:
    if not content.strip():
        st.warning("Paste some content to rewrite.")
        st.stop()
    if not chosen:
        st.warning("Pick at least one platform.")
        st.stop()

    selected = [platforms[key] for key in chosen]
    with st.spinner(f"Editing for {', '.join(p.name for p in selected)} with `{model}`..."):
        results = asyncio.run(run_platforms(content, selected, make_llm(model)))

    tabs = st.tabs([p.name for p in selected])
    for tab, platform, result in zip(tabs, selected, results, strict=True):
        with tab:
            if isinstance(result, Exception):
                st.error(f"{platform.name} failed: {result}")
                continue
            st.code(result.full_text, language=None, wrap_lines=True)
            st.caption(
                f"{result.character_count:,} / {platform.max_chars:,} characters · "
                f"{len(result.hashtags)} / {platform.max_hashtags} hashtags"
            )
            for warning in result.warnings:
                st.warning(warning)
            with st.expander("Critic's feedback", expanded=True):
                st.markdown("\n".join(f"- {point}" for point in result.critique))
