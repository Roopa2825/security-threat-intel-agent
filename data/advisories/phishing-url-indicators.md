# Phishing and Malicious URL Indicators - Analyst Guide

Lookalike domains (typosquatting): attackers register domains that differ from a real brand by one character, such as paypa1.com instead of paypal.com, or micros0ft.com instead of microsoft.com.

Homoglyphs: characters from other alphabets that look identical to Latin letters, for example a Cyrillic "а" in place of "a". These often appear in punycode form starting with xn--.

Brand name in the wrong place: a real brand used as a subdomain of an unrelated domain, for example paypal.com.secure-login.example.net, where the real domain is example.net.

Many subdomains or very long URLs: used to hide the real domain from the reader.

IP address instead of a domain: links such as http://192.0.2.10/login are a common phishing sign.

The @ symbol in a URL: everything before @ is ignored by the browser, so http://bank.com@evil.example goes to evil.example.

Suspicious words: login, verify, update, secure, account and password in the domain or path, especially combined with a brand name.

URL shorteners and newly registered domains: hide the destination or have no reputation history.

HTTPS is not proof of safety: attackers can obtain valid certificates for their own phishing domains.

Recommended action: do not click, report the message to the security team, block the domain at the email gateway and web proxy, and check whether any user already entered credentials.
