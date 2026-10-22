# ZCHSound access request (drafted 2026-10-21)

**Status of the public site.** The paper (Jia et al., IEEE TBME 71(8):2278-2286, 2024, CC BY 4.0) states that
the data "can be accessed and downloaded by the public at http://zchsound.ncrcch.org.cn/". On 2026-10-21 that
domain redirects to a Tencent Cloud ICP-filing block page over HTTP and fails over HTTPS; the Internet Archive's
last capture (2025-06-01) shows only an empty "Index Page". No mirror exists on Kaggle, Hugging Face, Zenodo or
GitHub. The only remaining route is the corresponding authors.

**Corresponding authors (from the paper's first-page footnote):** Weize Xu and Qiang Shu, Cardiac Center,
Children's Hospital, Zhejiang University School of Medicine, Hangzhou, China.
E-mail: weizexu@zju.edu.cn; shuqiang@zju.edu.cn.

---

**To:** weizexu@zju.edu.cn
**Cc:** shuqiang@zju.edu.cn
**Subject:** Access to the ZCHSound database

Dear Prof. Xu and Prof. Shu,

I am a researcher working on AI-based paediatric heart-sound screening for low-resource settings, and I would
like to use the ZCHSound database from your 2024 IEEE TBME paper as an external evaluation set.

The download site (http://zchsound.ncrcch.org.cn/) has been unreachable for some time. Could you let me know
whether it will be restored, or whether the clean and noisy subsets could be shared another way? I am happy to
sign a data-use agreement, and the data would be used only for non-commercial research and cited as your paper.

I would also be grateful for any advice you can offer on working with the data, for example on the recording
device, label conventions, or related work from your group that I should be aware of.

Thank you for your time and for making this resource available.

Kind regards,

Mustafa Shah
tariq@kentech.ac.kr

---

## Update 2026-10-22: data obtained

The first author publishes the dataset on GitHub: https://github.com/WeiJieOvO/ZCHSound-Dataset (created 2026-05-26,
README edited 2026-10-21, contact weijieovo69@gmail.com). `ZCHSound.zip` (339,554,986 bytes, Git LFS,
sha256 112432cb10c5ab768cabf217f0a090a20bd0102b093d8b42dbdabc19e39b58b7) was downloaded from the LFS endpoint,
the checksum verified, and the archive extracted to `data/raw/zchsound/extracted/ZCHSound/`:
941 clean recordings (NORMAL 533, VSD 187, ASD 119, PFO 70, PDA 32) and 318 noisy neonatal recordings, 8 kHz, 16-bit,
exactly as in the paper. Loader: `pcg/zchsound.py`; evaluation: `scripts/p4b_zchsound.py`.
Licence: the README says to follow the paper and the original site; cite Jia et al. 2024. Confirm the terms with the
authors if they reply; the e-mail above stays valid.
