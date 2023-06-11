
# Bạn muốn tham gia cùng xây dựng! Bây giờ bạn cần làm gì?

[Hãy tham gia vào Vietnamese LLMs Community Discord Server!](https://discord.gg/eH7eg4fT), đây là nơi bạn cỏ thể trao đổi và phối hợp công việc.

[Hoặc bạn có thể trực tiếp đưa ra issue và thảo luận ở phần Discussion ngạy tại github page](https://github.com/TranNhiem/Vietnamese_LLMs/discussions).

[Hoặc bạn có thể truy cập vào Notion Notebook để biết thêm về cấu trúc của dự án](https://scarce-cuticle-863.notion.site/Vietnamese-LLMs-Community-Project-aea639588fbe428480da15fd3e2a0c22?pvs=4)

### Tiếp nhận công việc

Chúng ta có một danh sách nhiệm vụ ngày càng phát triển của [vấn đề (issue)](https://github.com/TranNhiem/Vietnamese_LLMs/issues). Tìm một vấn đề hấp dẫn và để lại bình luận rằng bạn muốn làm việc trên nó. Trong bình luận của bạn, hãy mô tả ngắn gọn cách bạn sẽ giải quyết vấn đề và nếu có câu hỏi nào bạn muốn thảo luận. Khi một người phối hợp dự án đã gán vấn đề cho bạn, hãy bắt đầu làm việc trên nó.

Nếu vấn đề hiện tại chưa rõ ràng nhưng bạn quan tâm, vui lòng đăng lên Discord và ai đó sẽ giúp làm rõ vấn đề chi tiết hơn.

**Luôn luôn hoan nghênh:** Tài liệu định dạng markdown trong `docs/`, docstrings, sơ đồ kiến trúc hệ thống và tài liệu khác.

### Gửi công việc đã hoàn thành

Chúng tôi đang làm việc trên các phần khác nhau của Vietnamese_LLMs cùng nhau. Để đóng góp một cách suôn sẻ, có một số đề xuất thực hiện như sau:

1. [Fork (sao chép) dự án này](https://docs.github.com/en/get-started/quickstart/fork-a-repo) và clone (sao chép) nó xuống máy tính cục bộ của bạn. (Đọc thêm [về Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks))
2. Trước khi làm bất kỳ thay đổi nào, hãy thử [đồng bộ hóa kho dự án đã được forked](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) để giữ nó cập nhật với kho gốc (upstream repository).
3. Trên một [nhánh mới](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository) trong fork của bạn (có nghĩa là một "nhánh tính năng" và không phải là `main`), làm việc trên một thay đổi tập trung nhỏ chỉ liên quan đến một số tệp tin.
4. Chạy `pre-commit` và đảm bảo tất cả các tệp tin được định dạng đúng. Điều này giúp đơn giản hóa quá trình xem xét.
5. Đóng gói một phần nhỏ công việc giải quyết một phần của vấn đề [vào một yêu cầu kéo (Pull Request)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) và [gửi yêu cầu xem xét (review request)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review). [Đây là một ví dụ trích dẫn từ OpenAssistant về pull request](https://github.com/LAION-AI/Open-Assistant/pull/658) là một ví dụ về PR cho dự án này để minh họa quy trình này.
6. Nếu may mắn, chúng tôi có thể hợp nhất thay đổi của bạn vào `main` mà không gặp bất kỳ vấn đề nào. Nếu có thay đổi đối với tệp tin mà bạn đang làm việc, hãy giải quyết chúng bằng cách:
   1. Trước tiên, hãy thử rebase như được đề xuất [trong các hướng dẫn này](https://timwise.co.uk/2019/10/14/merge-vs-rebase/#should-you-rebase).
   2. Nếu rebase gây khó khăn quá, hãy merge như được đề xuất [trong các hướng dẫn này](https://timwise.co.uk/2019/10/14/merge-vs-rebase/#should-you-merge).
7. Sau khi bạn đã giải quyết xung đột (nếu có), hoàn thành quá trình xem xét và [squash và merge](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-commits) PR của bạn (khi squash, hãy cố gắng làm sạch hoặc cập nhật các thông điệp commit riêng lẻ để tạo ra một thông điệp đơn lẻ hợp lý).
8. Hợp nhất thay đổi của bạn và chuyển sang một vấn đề mới hoặc bước thứ hai của vấn đề hiện tại.

Ngoài ra, nếu ai đó đang làm việc trên một vấn đề mà bạn quan tâm, hãy hỏi xem họ cần sự trợ giúp hay muốn nhận ý kiến về cách tiếp cận vấn đề. Nếu có, hãy chia sẻ một cách tích cực. Nếu họ có vẻ đã giải quyết vấn đề một cách tốt, hãy để họ làm việc trên giải pháp của mình cho đến khi gặp khó khăn.

#### Mẹo

- Hãy cố gắng không làm việc trên nhánh `main` trong fork của bạn - lý tưởng là bạn có thể giữ nó chỉ là một bản sao đã được cập nhật của `main` từ `Vietnamese_LLMs`.
- Nếu nhánh tính năng của bạn bị lỗi, chỉ cần cập nhật nhánh `main` trong fork của bạn và tạo một nhánh tính năng mới sạch sẽ, trong đó bạn có thể thêm các thay đổi của mình một cách riêng biệt trong từng commit hoặc tất cả trong một commit duy nhất.

### Khi nào xem xét hoàn thành

Một quá trình xem xét được coi là hoàn thành khi tất cả các ý kiến chặn đã được giải quyết và ít nhất một người xem đã phê duyệt PR. Hãy đảm bảo nhận biết bất kỳ ý kiến không chặn nào bằng cách thực hiện thay đổi theo yêu cầu, giải thích tại sao nó không được giải quyết ngay bây giờ hoặc tạo một vấn đề để xử lý sau.


**Bất kỳ điều gì nằm trong cột `Todo` và không được gán, đều có sẵn cho ai muốn làm nhiệm vụ đó. Điều này có nghĩa là chúng tôi rất hoan nghênh bất kỳ ai thực hiện những nhiệm vụ này.**

Nếu bạn muốn làm việc trên một nhiệm vụ, hãy gán cho mình hoặc viết một bình luận cho biết bạn muốn làm việc trên nó và bạn định làm gì.

