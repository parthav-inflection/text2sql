"""
SQLFluff validation and refinement module for Text2SQL system.
This module validates and optionally fixes SQL queries using SQLFluff.
"""

import logging
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from src.modules.base import RefinementModule

if TYPE_CHECKING:
    from src.agents.base import AgentContext

logger = logging.getLogger(__name__)


class SQLFluffValidatorModule(RefinementModule):
    """Module for validating and refining SQL queries using SQLFluff."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # SQLFluff configuration
        self.dialect = self.module_config.get('dialect', 'sqlite')
        self.auto_fix = self.module_config.get('auto_fix', True)
        self.fail_on_parse_error = self.module_config.get('fail_on_parse_error', True)
        
        # Initialize SQLFluff
        self.sqlfluff_available = False
        try:
            import sqlfluff
            self.sqlfluff = sqlfluff
            self.sqlfluff_available = True
            logger.info(f"SQLFluff module initialized with dialect: {self.dialect}")
        except ImportError:
            logger.warning("SQLFluff not available, module will be disabled")
            self.enabled = False
    
    def validate_config(self):
        """Validate module configuration."""
        super().validate_config()
        
        if not self.sqlfluff_available:
            logger.warning("SQLFluff module disabled due to missing dependency")
            return
        
        # Validate dialect
        try:
            available_dialects = [d.label for d in self.sqlfluff.list_dialects()]
            if self.dialect not in available_dialects:
                logger.warning(f"Dialect '{self.dialect}' not available. Available: {available_dialects}")
        except Exception as e:
            logger.warning(f"Could not validate dialect: {e}")
    
    def validate_and_fix_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query using SQLFluff and optionally fix it.
        
        Returns:
            Dict with validation and fixing results
        """
        if not self.sqlfluff_available or not self.enabled:
            return {
                'valid': True,
                'original_sql': sql_query,
                'fixed_sql': sql_query,
                'errors': [],
                'violations': [],
                'module_applied': False
            }
        
        try:
            # Lint the SQL to find violations
            violations = self.sqlfluff.lint(sql_query, dialect=self.dialect)
            
            # Check if there are any parsing or syntax errors
            critical_violations = [
                v for v in violations 
                if v.get('code', '').startswith('PRS') or  # Parse errors
                   v.get('code', '').startswith('TMP') or  # Template errors
                   'syntax error' in v.get('description', '').lower()
            ]
            
            is_valid = len(critical_violations) == 0
            error_messages = []
            
            for violation in critical_violations:
                error_msg = f"Line {violation.get('line_no', '?')}: {violation.get('description', 'Unknown error')}"
                error_messages.append(error_msg)
            
            # Try to fix the SQL if enabled and there are violations
            fixed_sql = sql_query
            auto_fix_applied = False
            
            if self.auto_fix and violations:
                try:
                    fixed_sql = self.sqlfluff.fix(sql_query, dialect=self.dialect)
                    # Remove any trailing newlines
                    fixed_sql = fixed_sql.strip()
                    
                    # If fix didn't change anything, keep original
                    if fixed_sql == sql_query:
                        fixed_sql = sql_query
                    else:
                        auto_fix_applied = True
                        logger.info(f"SQLFluff auto-fixed SQL query")
                        logger.debug(f"Original: {sql_query}")
                        logger.debug(f"Fixed: {fixed_sql}")
                        
                        # Re-validate the fixed SQL to check if parsing errors were resolved
                        fixed_violations = self.sqlfluff.lint(fixed_sql, dialect=self.dialect)
                        fixed_critical_violations = [
                            v for v in fixed_violations 
                            if v.get('code', '').startswith('PRS') or 
                               v.get('code', '').startswith('TMP') or
                               'syntax error' in v.get('description', '').lower()
                        ]
                        
                        # If auto-fix resolved the parsing errors, don't report them
                        if len(fixed_critical_violations) == 0:
                            is_valid = True
                            error_messages = []  # Clear errors since they were fixed
                        else:
                            # Auto-fix didn't resolve all parsing errors
                            error_messages = []
                            for violation in fixed_critical_violations:
                                error_msg = f"Line {violation.get('line_no', '?')}: {violation.get('description', 'Unknown error')}"
                                error_messages.append(error_msg)
                        
                except Exception as fix_error:
                    logger.warning(f"SQLFluff auto-fix failed: {fix_error}")
                    fixed_sql = sql_query
            
            return {
                'valid': is_valid,
                'original_sql': sql_query,
                'fixed_sql': fixed_sql,
                'errors': error_messages,  # Only contains unresolved parsing errors
                'violations': violations,  # Original violations for debugging
                'auto_fix_applied': auto_fix_applied,
                'module_applied': True
            }
            
        except Exception as e:
            logger.error(f"SQLFluff validation failed: {e}")
            # Fall back to assuming valid if SQLFluff fails
            return {
                'valid': True,
                'original_sql': sql_query,
                'fixed_sql': sql_query,
                'errors': [f"SQLFluff validation error: {str(e)}"],
                'violations': [],
                'module_applied': False
            }
    
    def refine_sql(self, sql: str, context: 'AgentContext') -> str:
        """
        Refine/correct the SQL query using SQLFluff.
        This is the main interface called by the module pipeline.
        """
        validation_result = self.validate_and_fix_sql(sql)
        
        # Store validation results in context for later use
        context.set_module_output(self.name, validation_result)
        
        # Return the fixed SQL if auto-fix is enabled and valid
        if self.auto_fix and validation_result['valid']:
            return validation_result['fixed_sql']
        elif not validation_result['valid'] and self.fail_on_parse_error:
            # If parse error and configured to fail, keep original to let execution handle the error
            return validation_result['original_sql']
        else:
            return validation_result['fixed_sql']
    
    def process(self, context: 'AgentContext'):
        """
        Process method for the module pipeline.
        
        Note: This module doesn't fit perfectly into the current tool-calling pipeline
        since it operates on individual SQL queries rather than the entire context.
        It stores its validation function in the context for the SQLExecutionTool to use.
        """
        if not self.enabled or not self.sqlfluff_available:
            return
        
        # Store a reference to this module's validation function in the context
        # so the SQLExecutionTool can use it
        context.set_module_output('sqlfluff_validator_fn', self.validate_and_fix_sql)
        logger.debug(f"SQLFluff validator module registered in context")
        
    def get_validation_stats(self, context: 'AgentContext') -> Dict[str, Any]:
        """Get statistics about SQLFluff validation for this context."""
        validation_results = []
        
        # Collect all validation results stored during this session
        for key, value in context.module_outputs.items():
            if key.startswith(self.name) and isinstance(value, dict) and 'module_applied' in value:
                validation_results.append(value)
        
        if not validation_results:
            return {'total_queries': 0, 'valid_queries': 0, 'auto_fixed_queries': 0}
        
        total_queries = len(validation_results)
        valid_queries = sum(1 for r in validation_results if r.get('valid', True))
        auto_fixed_queries = sum(1 for r in validation_results 
                               if r.get('original_sql') != r.get('fixed_sql'))
        
        return {
            'total_queries': total_queries,
            'valid_queries': valid_queries,
            'auto_fixed_queries': auto_fixed_queries,
            'parsing_success_rate': valid_queries / total_queries if total_queries > 0 else 0.0,
            'auto_fix_rate': auto_fixed_queries / total_queries if total_queries > 0 else 0.0
        } 