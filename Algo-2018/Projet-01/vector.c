#include "vector.h"

acceleration_vector* create_acceleration_vector( void )
{
	acceleration_vector* new_av = malloc(sizeof(acceleration_vector));
	if ( new_av == NULL ) return NULL;
	new_av->acceleration_x = 0.0;
	new_av->acceleration_y = 0.0;

	return new_av;
}

speed_vector* create_speed_vector( void )
{
	speed_vector* new_sv = malloc(sizeof(speed_vector));
	if ( new_sv == NULL ) return NULL;
	new_sv->speed_x = 0.0;
	new_sv->speed_y = 0.0;

	return new_sv;
}

void free_acceleration_vector( acceleration_vector* av )
{
	if ( av != NULL ) free( av );
}

void free_speed_vector( speed_vector* sv )
{
	if ( sv != NULL ) free( sv );
}

void acceleration_vector_sum( acceleration_vector* av1, acceleration_vector* av2 )
{
	av1->acceleration_x += av2->acceleration_x;
	av1->acceleration_y += av2->acceleration_y;
}

void speed_vector_sum( speed_vector* sv1, speed_vector* sv2 )
{
	sv1->speed_x += sv2->speed_x;
	sv1->speed_y += sv2->speed_y;
}

void apply_acceleration( acceleration_vector* av, speed_vector* sv )
{
	sv->speed_x += av->acceleration_x;
	sv->speed_y += av->acceleration_y;
}

void reset_acceleration_vector( acceleration_vector* sv )
{
	av1->acceleration_x = 0.0;
	av1->acceleration_y = 0.0;
}

void reset_speed_vector( speed_vector* sv );
{
	sv->speed_x = 0.0;
	sv->speed_y = 0.0;
}