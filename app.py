import streamlit as st
from scorer.quality import score_quality
from scorer.toxicity import score_toxicity

st.set_page_config(page_title="LLM Text Scoring", layout="centered")
st.title("LLM Text Quality and Toxicity Analysis")

text_input = st.text_area("Please enter some text:")

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            quality = score_quality(text_input)
            toxicity = score_toxicity(text_input)

        st.success("✔️ Analysis Completed")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🟩 Quality")
            st.metric("Quality Score", f"{quality['quality_score']} / 100")
            st.metric("Readability", f"{quality['readability']:.1f}")
            st.write(f"Grammar Errors: {quality['error_count']}")
            if quality['suggestions']:
                st.markdown("**Suggestions:**")
                for s in quality['suggestions'][:3]:
                    st.write(f"• {s}")

        with col2:
            st.markdown("### 🟥 Toxicity")
            st.metric("Toxicity Score", toxicity['toxic_score'])
            st.write(f"Label: **{toxicity['label']}**")
            st.info(toxicity['explanation'])

            
            if toxicity.get("detected_toxic_phrases"):
                st.markdown("**Detected Toxic Phrase(s) by Model:**")
                for phrase in toxicity["detected_toxic_phrases"]:
                    st.write(f"> {phrase}")

        
        advice = []
        if quality['quality_score'] < 80:
            advice.append("• Improve grammar or spelling.")
        if quality['readability'] < 60:
            advice.append("• Simplify your sentences for better readability.")
        if toxicity['toxic_score'] > 0.3:
            advice.append("• Consider removing harmful language.")

        if advice:
            st.markdown("Improvement Advice")
            for line in advice:
                st.write(line)
        else:
            st.markdown("Your text looks good!")
