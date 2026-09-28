"""CLI --format csv: one row per document, the fields of AgenticDownload as columns."""
import csv
import io

from docseek.__main__ import write_csv
from docseek.models import AgenticDownload


def test_one_row_per_document_with_every_field():
    downloads = [
        AgenticDownload(url='https://site.example/a.pdf', name='Annual report, 2025', reason='', source_page='https://site.example/',
                        relevance=0.91, verdict='accepted', source='page', year='2025'),
        AgenticDownload(url='https://site.example/b.pdf', name='Havi "jelentés"', reason='', source_page='https://site.example/'),
    ]
    out = io.StringIO()
    write_csv(downloads, out)
    rows = list(csv.DictReader(io.StringIO(out.getvalue())))
    assert list(rows[0]) == list(AgenticDownload.model_fields)
    assert rows[0]['name'] == 'Annual report, 2025' and rows[0]['relevance'] == '0.91' and rows[0]['year'] == '2025'
    assert rows[1]['name'] == 'Havi "jelentés"' and rows[1]['verdict'] == 'unscored' and rows[1]['relevance'] == ''


def test_no_documents_is_just_the_header():
    out = io.StringIO()
    write_csv([], out)
    assert out.getvalue() == ','.join(AgenticDownload.model_fields) + '\n'
