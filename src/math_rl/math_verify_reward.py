"""Pinned Math-Verify candidate. Extraction never receives the reference answer."""
from dataclasses import asdict
from importlib.metadata import version

RULE_VERSION = 'math-verify-v1'
PINS = {'math-verify': '0.9.0', 'latex2sympy2_extended': '1.11.0',
        'antlr4-python3-runtime': '4.13.2', 'sympy': '1.14.0', 'mpmath': '1.3.0'}
TIMEOUT = 3
MAX_CHARS = 32000
PARSE_ERROR_POLICY = 'SympifyError during extraction is parse_failure; other exceptions remain fatal in batch scoring'
TIMEOUT_POLICY = 'Per-answer parse_timeout and verify_timeout receive reward 0 with status preserved; not proven incorrect'


def configuration():
    from math_verify import LatexExtractionConfig, ExprExtractionConfig
    from latex2sympy2_extended import NormalizationConfig
    # Follow reward-oriented normalization: do not repair malformed operators.
    latex = LatexExtractionConfig(boxed_match_priority=0, try_extract_without_anchor=True,
        normalization_config=NormalizationConfig(basic_latex=True, units=True,
            malformed_operators=False, nits=False, boxed='all', equations=False))
    return [latex, ExprExtractionConfig(try_extract_without_anchor=True)]


def provenance():
    installed = {name: version(name) for name in PINS}
    if installed != PINS:
        raise RuntimeError('Verifier package versions differ; install requirements-verifier.txt in an isolated environment')
    return dict(packages=installed, extraction=[asdict(x) for x in configuration()],
                fallback_mode='no_fallback', extraction_mode='first_match',
                timeout_seconds=TIMEOUT, max_response_chars=MAX_CHARS,
                parse_error_policy=PARSE_ERROR_POLICY,
                timeout_policy=TIMEOUT_POLICY,
                float_rounding=6, numeric_precision=15, strict=True,
                selection='Library priority order, boxed first, first match only; no reference-guided search',
                limitations='May select an unrelated or contradicted expression, including an unanchored number. Human audit required.')


def grade(response, ground_truth):
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException
    from sympy import Rational, SympifyError
    from math_rl.reward import parse_number
    # GSM8K references are validated exact decimals, not arbitrary expressions.
    target = parse_number(ground_truth)
    if target is None:
        raise ValueError(f'Invalid reference: {ground_truth!r}')
    gold = Rational(str(target))
    result = dict(extracted=None, diagnostic_reward=0, status='parse_failure')
    if len(response) > MAX_CHARS:
        return dict(result, status='input_too_long')
    try:
        prediction = parse(response, extraction_config=configuration(),
                           fallback_mode='no_fallback', extraction_mode='first_match',
                           parsing_timeout=TIMEOUT, raise_on_error=True)
    except TimeoutException:
        return dict(result, status='parse_timeout')
    except SympifyError as exc:
        # An expression could not be converted by SymPy: no extracted answer.
        # Handle only this known conversion failure, not arbitrary runtime errors.
        return dict(result, error=type(exc).__name__)
    except (MemoryError, OSError, ImportError):
        raise
    except Exception as exc:
        return dict(result, status='parse_error', error=type(exc).__name__)
    if not prediction:
        return result
    # No fallback strings or search through multiple candidates against gold.
    if len(prediction) != 1 or isinstance(prediction[0], str):
        return dict(result, status='unsupported_parse')
    result['extracted'] = str(prediction[0])
    try:
        correct = verify(gold, prediction[0], float_rounding=6, numeric_precision=15,
                         strict=True, timeout_seconds=TIMEOUT, raise_on_error=True)
    except TimeoutException:
        return dict(result, status='verify_timeout')
    except (MemoryError, OSError, ImportError):
        raise
    except Exception as exc:
        return dict(result, status='verify_error', error=type(exc).__name__)
    return dict(result, diagnostic_reward=int(correct), status='correct' if correct else 'incorrect')
