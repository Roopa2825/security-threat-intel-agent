from app.url_analyzer import analyze_url, edit_distance, is_valid_cve_id


def test_real_brand_domain_is_low_risk():
    assert analyze_url("https://www.paypal.com/signin")["verdict"] != "high risk"


def test_typosquatting_is_flagged():
    result = analyze_url("http://paypa1.com/login/verify")
    assert result["verdict"] == "high risk"
    assert any("typosquatting" in r for r in result["reasons"])


def test_brand_in_subdomain_is_flagged():
    result = analyze_url("https://paypal.com.secure-login.example.net/account")
    assert result["risk_score"] >= 30
    assert any("real domain is example.net" in r for r in result["reasons"])


def test_at_symbol_and_ip_address():
    assert analyze_url("http://bank.com@192.0.2.10/login")["verdict"] == "high risk"


def test_plain_safe_url():
    assert analyze_url("https://www.asu.edu")["verdict"] == "low risk"


def test_edit_distance():
    assert edit_distance("paypa1", "paypal") == 1


def test_cve_id_validation():
    assert is_valid_cve_id("CVE-2021-44228")
    assert not is_valid_cve_id("log4shell")


def test_lookalike_characters_with_extra_words():
    result = analyze_url("https://micros0ft-support.xyz/update")
    assert result["verdict"] == "high risk"
