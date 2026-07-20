# Tracked format fixtures

このディレクトリのfixtureは、リファクタリング中にbinary formatや探索意味が意図せず変わることを検出するためGitで追跡する。

- `halfkp/`: HalfKP v1 headerの正確なbyte列
- `teacher/`: HKST v2 headerと最小1-record datasetの正確なbyte列
- `search/`: 決定的search fingerprintへ入力する局面

Binaryを直接編集しにくいため、format fixtureは空白区切りの16進byte列として保存する。
テストはこれをbyte列へ戻し、reader/writerの両方向を検証する。

fixtureを更新する場合は、formatまたはsemantics versionを変更する必要があるか先に確認し、旧fixtureを理由なく上書きしない。
