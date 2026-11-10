import fitz

from chunk_sds import chunk_sds


def test_extracts_sections_and_tracks_pages(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf = fitz.open()
    page_one = pdf.new_page()
    page_one.insert_textbox((60, 60, 540, 180), "SECTION 1: IDENTIFICATION\nProduct name: Sample solvent")
    page_two = pdf.new_page()
    page_two.insert_textbox((60, 60, 540, 180), "Section 2: Hazard(s) Identification\nCauses eye irritation.")
    pdf.save(pdf_path)
    pdf.close()

    chunks = chunk_sds(str(pdf_path))

    assert [chunk["section_number"] for chunk in chunks] == [1, 2]
    assert chunks[0]["page_start"] == 1
    assert chunks[1]["page_start"] == 2
    assert chunks[1]["section_title"] == "Hazard(s) Identification"
