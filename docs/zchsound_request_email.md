# ZCHSound access request (drafted 2026-10-21)

**Status of the public site.** The paper (Jia et al., IEEE TBME 71(8):2278-2286, 2024, CC BY 4.0) states that
the data "can be accessed and downloaded by the public at http://zchsound.ncrcch.org.cn/". On 2026-10-21 that
domain redirects to a Tencent Cloud "备案" (ICP filing) block page over HTTP and fails over HTTPS; the Internet
Archive's last capture (2025-06-01) shows only an empty "Index Page". No mirror exists on Kaggle, Hugging Face,
Zenodo or GitHub. The only remaining route is the corresponding authors.

**Corresponding authors (from the paper's first-page footnote):** Weize Xu and Qiang Shu, Department of Cardiac
Center, Children's Hospital, Zhejiang University School of Medicine, National Clinical Research Center for Child
Health, Hangzhou 310020, China. E-mail: weizexu@zju.edu.cn; shuqiang@zju.edu.cn.

---

**To:** weizexu@zju.edu.cn
**Cc:** shuqiang@zju.edu.cn
**Subject:** Request for access to the ZCHSound paediatric heart sound database (download site unreachable)

Dear Prof. Xu and Prof. Shu,

I am writing to request access to the ZCHSound database described in your paper "ZCHSound: Open-Source ZJU
Paediatric Heart Sound Database With Congenital Heart Disease" (IEEE Trans. Biomed. Eng., 2024).

The paper states that the data can be downloaded from http://zchsound.ncrcch.org.cn/. As of 21 October 2026
that address returns a hosting-provider block page over HTTP and does not respond over HTTPS, and the Internet
Archive holds no usable copy, so I have not been able to obtain the files.

I am an engineer working on resource-efficient and trustworthy AI for paediatric heart-sound screening, with the
goal of a peer-reviewed paper and a PhD application. My current study trains murmur classifiers on the CirCor
DigiScope dataset and audits how INT8/INT4 quantization for low-cost stethoscope devices changes their
subgroup performance, calibration and explanations. ZCHSound is the only public paediatric set recorded on a
different stethoscope with echocardiography-confirmed labels and a separate noisy subset, which makes it the
ideal external test set for device and population shift. The data would be used strictly for this
non-commercial research, stored on my own machines, not redistributed, and cited as your TBME paper in any
publication.

Could you please let me know whether the download site will be restored, or whether the database (ideally
both the clean and the noisy subsets with their labels) can be shared by another means? I am happy to sign a
data-use agreement if one is required.

Thank you very much for making this resource available to the community.

Kind regards,

Mustafa Shah
tariq@kentech.ac.kr
